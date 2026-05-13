# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import collections
import inspect
import os
import queue
import random
import secrets
import sys
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import numpy as np
import sentencepiece  # type: ignore[import-untyped]
import sphn  # type: ignore[import-untyped]
import torch
from aiohttp import web
from google.cloud import speech
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from ._tar_utils import extract_data_archive
from .client_utils import log
from .models import LMGen, LMModel, MimiModel, loaders
from .run_inference import get_condition_tensors

# -----------------------
# Thread-safe conversation state
# -----------------------
conversation_text = ""
current_speaker = None
conversation_lock = threading.Lock()

save_dir: Path | None = None


def configure_save_dir(log_dir: str | None) -> None:
    global save_dir
    if not log_dir:
        save_dir = None
        return
    save_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log("info", f"Plaintext session logging enabled at {save_dir}")


def append_session_log(filename: str, text: str) -> None:
    if save_dir is None:
        return
    with (save_dir / filename).open("a", encoding="utf-8") as f:
        f.write(text)


def clear_session_logs() -> None:
    if save_dir is None:
        return
    for filename in [
        "asr_words.txt",
        "conversation.txt",
        "llm_stream_words.txt",
        "moshi_words.txt",
        "oracle_stream.txt",
    ]:
        fpath = save_dir / filename
        if fpath.exists():
            fpath.unlink()
    log("info", f"Cleared session log files in {save_dir}")


def add_to_conversation(speaker: str, text: str, flush_file: bool = True):
    """Append text to the conversation in a thread-safe way."""
    global conversation_text, current_speaker
    text = (text or "").strip()
    if not text:
        return
    with conversation_lock:
        if speaker != current_speaker:
            if conversation_text and not conversation_text.endswith("\n"):
                conversation_text += "\n"
            conversation_text += f"{speaker}: "
            current_speaker = speaker
        conversation_text += f"{text} "
        if flush_file and save_dir is not None:
            (save_dir / "conversation.txt").write_text(conversation_text, encoding="utf-8")


def get_conversation_snapshot() -> str:
    with conversation_lock:
        return conversation_text


def get_last_speaker(conversation_snapshot: str) -> str | None:
    lines = conversation_snapshot.strip().split("\n")

    for line in reversed(lines):
        line = line.strip()
        if line and ":" in line:
            speaker = line.split(":", 1)[0].strip()
            return speaker

    return None


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# -----------------------
# ASR Processor (commit-pointer + short TTL dedupe)
# -----------------------
class AsyncASRProcessor:
    """Google Cloud Speech streaming with partial commits (no external final).
    - Monotonic commit-pointer per utterance to avoid duplicates on backtrack.
    - Short-term dedupe across final boundaries (TTL ~1.2s).
    - Logs each committed word to asr_words.txt with timestamp.
    """

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000  # Google Speech API requirement

        self.audio_buffer = queue.Queue(maxsize=100)
        self.running = False
        self.asr_task: asyncio.Task | None = None

        self.stats = {"words_detected": 0, "final_transcripts": 0, "buffer_drops": 0, "reconnections": 0}

        self.asr_enabled = False
        self.init_error: str | None = None
        self.speech_client: speech.SpeechClient | None = None
        self.config = None
        self.streaming_config = None

        # per-utterance commit pointer
        self._committed_len = 0

        # recent dedupe (word -> last_seen_ts), kept as deque for easy purge
        self._recent_words = collections.deque(maxlen=512)
        self._dedupe_ttl_sec = 1.2

        # nudgers
        self._nudge_partial = None  # def (full_text: str) -> None
        self._nudge_force = None  # def () -> None

        self._initialize_speech_client()

    def set_llm_nudgers(self, nudge_partial, nudge_force):
        """Register minimal nudgers for LLM restarts."""
        self._nudge_partial = nudge_partial
        self._nudge_force = nudge_force

    def _initialize_speech_client(self):
        try:
            if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                self.init_error = "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
                return
            self.speech_client = speech.SpeechClient()
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.target_sample_rate,
                language_code="en-US",
                enable_automatic_punctuation=False,
                enable_word_time_offsets=False,
                enable_word_confidence=False,
                use_enhanced=True,
                metadata=speech.RecognitionMetadata(
                    interaction_type=speech.RecognitionMetadata.InteractionType.VOICE_SEARCH,
                    microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                    recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PC,
                ),
            )
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=True,
                single_utterance=False,
            )
            self.asr_enabled = True
            self.init_error = None
            log("info", "Async ASR processor initialized")
        except Exception as e:
            self.init_error = str(e)
            log("warning", f"ASR initialization failed: {e}")

    async def start(self):
        if self.asr_enabled and not self.running:
            self.running = True
            self.asr_task = asyncio.create_task(self._run_asr_streaming())
            log("info", "Async ASR streaming started")

    async def stop(self):
        if self.running:
            self.running = False
            try:
                self.audio_buffer.put(None, block=False)
            except queue.Full:
                pass

            if self.asr_task:
                self.asr_task.cancel()
                try:
                    await self.asr_task
                except asyncio.CancelledError:
                    pass
            log("info", f"Async ASR streaming stopped. Stats: {self.stats}")

    def process_audio(self, pcm_data: np.ndarray):
        """Accept float32 [-1,1] PCM (or int16). Downsample to 16 kHz and enqueue LINEAR16 bytes."""
        if not self.asr_enabled or not self.running:
            return
        try:
            pcm_16bit: np.ndarray
            if isinstance(pcm_data, np.ndarray):
                if pcm_data.dtype == np.float32:
                    pcm_data = np.clip(pcm_data, -1.0, 1.0)
                    pcm_16bit = (pcm_data * 32767).astype(np.int16)
                else:
                    pcm_16bit = pcm_data.astype(np.int16)
            else:
                pcm_float = pcm_data.numpy() if hasattr(pcm_data, "numpy") else np.asarray(pcm_data, dtype=np.float32)
                pcm_float = np.clip(pcm_float, -1.0, 1.0)
                pcm_16bit = (pcm_float * 32767).astype(np.int16)

            # naive resample to 16k
            if self.sample_rate == 24000:
                idx: np.ndarray = np.arange(0, len(pcm_16bit), 1.5).astype(int)
                pcm_16k = pcm_16bit[idx[: min(len(idx), len(pcm_16bit))]]
            elif self.sample_rate == 16000:
                pcm_16k = pcm_16bit
            else:
                ratio = self.target_sample_rate / float(self.sample_rate)
                idx = np.arange(0, len(pcm_16bit), 1 / ratio).astype(int)
                pcm_16k = pcm_16bit[idx[: min(len(idx), len(pcm_16bit))]]

            try:
                self.audio_buffer.put(pcm_16k.tobytes(), block=False)
            except queue.Full:
                self.stats["buffer_drops"] += 1
                try:
                    _ = self.audio_buffer.get_nowait()
                    self.audio_buffer.put(pcm_16k.tobytes(), block=False)
                except Exception:
                    pass
        except Exception as e:
            log("error", f"Error processing audio: {e}")

    async def _run_asr_streaming(self):
        retry_count = 0
        max_retries = 5
        while self.running:
            try:
                await asyncio.to_thread(self._run_speech_streaming_once)
                retry_count = 0
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                log("info", "ASR streaming cancelled")
                raise
            except Exception as e:
                if self.running:
                    retry_count += 1
                    self.stats["reconnections"] += 1
                    log("error", f"ASR streaming error (retry {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        log("warning", "Max retries reached, waiting before reset...")
                        await asyncio.sleep(30)
                        retry_count = 0
                    else:
                        await asyncio.sleep(min(retry_count * 0.5, 2.0))

    def _run_speech_streaming_once(self):
        assert self.speech_client is not None
        max_duration = 240  # seconds

        def audio_generator():
            min_chunk_bytes = 1600  # 100ms @16k
            start_time = time.time()
            last_data_time = time.time()
            while self.running and (time.time() - start_time) < max_duration:
                chunks = []
                total = 0
                try:
                    chunk = self.audio_buffer.get(timeout=0.1)
                    if chunk is None:
                        return
                    chunks.append(chunk)
                    total += len(chunk)
                    last_data_time = time.time()
                except queue.Empty:
                    if time.time() - last_data_time > 5.0:
                        # keep stream alive with short silence
                        yield b"\x00" * 160
                        last_data_time = time.time()
                    continue

                deadline = time.time() + 0.05
                while total < min_chunk_bytes and time.time() < deadline:
                    try:
                        more = self.audio_buffer.get(timeout=0.01)
                        if more is None:
                            return
                        chunks.append(more)
                        total += len(more)
                    except queue.Empty:
                        break

                while True:
                    try:
                        more = self.audio_buffer.get_nowait()
                        if more is None:
                            return
                        chunks.append(more)
                    except queue.Empty:
                        break

                if chunks:
                    yield b"".join(chunks)

        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator())
        responses = self.speech_client.streaming_recognize(self.streaming_config, requests)
        self._process_responses(responses)

    def _purge_recent(self, now_ts):
        # purge old
        while self._recent_words and (now_ts - self._recent_words[0][1]) > self._dedupe_ttl_sec:
            self._recent_words.popleft()

    def _filter_dedupe(self, tokens):
        """Drop tokens that appeared in recent TTL window (case-insensitive)."""
        if not tokens:
            return tokens
        now_ts = time.time()
        self._purge_recent(now_ts)
        recent_set = {w for (w, ts) in self._recent_words if (now_ts - ts) <= self._dedupe_ttl_sec}
        out = []
        for w in tokens:
            wl = w.lower()
            if wl not in recent_set:
                out.append(w)
                self._recent_words.append((wl, now_ts))
        return out

    def _process_responses(self, responses):
        """Monotonic commits per result + short TTL dedupe across results."""
        for response in responses:
            if not self.running:
                break
            if not response.results:
                continue

            for result in response.results:
                if not result.alternatives:
                    continue

                transcript = (result.alternatives[0].transcript or "").strip()
                tokens = transcript.split()

                # growth beyond committed pointer
                first_commit = self._committed_len == 0
                if len(tokens) > self._committed_len:
                    new_tokens = tokens[self._committed_len :]
                    # short-term dedupe (across result boundaries)
                    commit_tokens = self._filter_dedupe(new_tokens)

                    if commit_tokens:
                        add_to_conversation("user", " ".join(commit_tokens), flush_file=True)
                        for w in commit_tokens:
                            self.stats["words_detected"] += 1
                            ts = int(time.time() * 1000)
                            log("info", f"[ASR Word] {w}")
                            append_session_log("asr_words.txt", f"{ts}: {w}\n")

                    # move pointer to current tokens length regardless of dedupe
                    self._committed_len = len(tokens)

                    # On first commit of an utterance: force-start LLM immediately.
                    if first_commit and self._nudge_force:
                        try:
                            self._nudge_force()
                        except Exception:
                            pass
                # Nudge LLM on every partial (debounced inside mux).
                if self._nudge_partial:
                    try:
                        self._nudge_partial(transcript)
                    except Exception:
                        pass

                if result.is_final:
                    self._committed_len = 0
                    self.stats["final_transcripts"] += 1


# -----------------------
# LLM Stream (parallel starts, single adopted stream)
# -----------------------
class LLMStreamMultiplexer:
    """
    Multiplex LLM streaming with *emission-based adoption*.

    Key policies:
    - Bounded number of live LLM streams (max_concurrent_streams).
    - A new stream is *not* adopted at start-time. It becomes the active stream
      only when it emits its first token ("adopt on first token").
    - Upon adoption:
        * Flush pending LLM events so old chunks cannot apply after the switch.
        * Cancel all strictly older generations (both never-emitted and mid-stream).
        * Keep strictly newer generations that have not emitted yet as warm standbys.
    - The single writer (server_state.llm_event_queue consumer) remains the only
      place where LMGen is actually fed, so ordering stays consistent.

    Simplifications in this version:
    - No debounce.
    - No "3 per 2 seconds" restart cap.
    - No delta-based nudging (min_chars_delta / min_words_delta are ignored).
    - Streams are started at a fixed cadence gated only by min_restart_interval.
    """

    def __init__(
        self,
        server_state,
        system_prompt: str = "",
        *,
        min_chars_delta: int = 1,  # unused (kept for API compatibility)
        min_words_delta: int = 1,  # unused (kept for API compatibility)
        min_restart_interval: float = 0.4,
        debounce_seconds: float = 0.0,  # unused (kept for API compatibility)
        max_restarts_per_2s: int = 0,  # unused (kept for API compatibility)
        max_prompt_chars: int = 6000,
        max_concurrent_streams: int = 20,
    ):
        self.server_state = server_state
        self.system_prompt = system_prompt
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before starting the server to enable LLM streaming."
            )
        self.client = AsyncOpenAI()

        # Fixed-cadence start guard
        self.min_restart_interval = float(min_restart_interval)
        self.max_prompt_chars = max_prompt_chars
        self.max_concurrent_streams = max(1, int(max_concurrent_streams))

        # loop / scheduling
        self.loop: asyncio.AbstractEventLoop | None = None
        self._last_start_ts = 0.0

        # generation bookkeeping
        self._gen_counter = 0
        self._tasks: dict[int, asyncio.Task] = {}  # gen_id -> task

        # Emission-based adoption:
        # - adopted_gen: only this gen's chunks are forwarded to the writer
        # - _first_emit_ts: first token time per gen (0.0/absent => not yet emitted)
        # - _start_ts: start time per gen (for TTFT logging / debugging)
        self.adopted_gen: int = 0
        self._first_emit_ts: dict[int, float] = {}
        self._start_ts: dict[int, float] = {}

        # Backward-compat shadow of "latest" (equals adopted_gen after adoption)
        self._latest_gen: int = 0

        # Prevent concurrent adoption races
        self._adopt_lock = asyncio.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    # called from any thread / callback
    def on_interim_pending(self, _full_text: str):
        """No debounce, no delta gating. Just try to start at fixed cadence."""
        loop = self.loop
        if not loop:
            return
        loop.call_soon_threadsafe(lambda: loop.create_task(self._maybe_start_new_stream(force=False)))

    # called on first commit within an utterance
    def nudge_now(self, force: bool = True):
        """Force-start immediately (bypasses the cadence guard)."""
        loop = self.loop
        if not loop:
            return
        loop.call_soon_threadsafe(lambda: loop.create_task(self._maybe_start_new_stream(force=force)))

    def _trim_prompt(self, text: str) -> str:
        if len(text) <= self.max_prompt_chars:
            return text
        return text[-self.max_prompt_chars :]

    def _build_messages(self, committed_conversation: str):
        convo = committed_conversation.rstrip()
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._trim_prompt(convo)},
        ]

    async def _maybe_start_new_stream(self, *, force: bool):
        """
        Fixed-cadence start policy:
        - If 'force' is False, start only when at least 'min_restart_interval' has elapsed.
        - If 'force' is True, start unconditionally (used for first-commit kick).
        """
        now = time.time()
        if not force and (now - self._last_start_ts) < self.min_restart_interval:
            return

        committed = get_conversation_snapshot()
        last_speaker = get_last_speaker(committed)
        if last_speaker != "user":
            return
        await self._start_stream(committed)

        self._last_start_ts = time.time()

    async def _start_stream(self, committed_conversation: str):
        assert self.loop is not None

        # 1) assign a new generation id
        self._gen_counter += 1
        gen_id = self._gen_counter

        # record timings and mark "not yet emitted"
        self._start_ts[gen_id] = time.time()
        self._first_emit_ts[gen_id] = 0.0  # 0.0 => has not emitted a token yet

        # NOTE: we DO NOT adopt here. Adoption happens on the first emitted token.

        # 2) start the new stream
        messages = self._build_messages(committed_conversation)
        task = self.loop.create_task(self._stream_single(messages, gen_id))
        self._tasks[gen_id] = task
        log("info", f"LLM started (gen {gen_id})")

        # 3) enforce max_concurrent_streams (cancel surplus, but never the adopted gen)
        await self._enforce_stream_limit()

    async def _adopt_generation(self, gen_id: int):
        """Adopt 'gen_id' as the active speaker when it emits its first token."""
        async with self._adopt_lock:
            if gen_id == self.adopted_gen:
                return  # already adopted

            # 1) flush pending writer events (avoid late old chunks after switch)
            try:
                while True:
                    _ = self.server_state.llm_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            # 2) cancel all strictly older gens (both never-emitted and mid-stream)
            for gid, t in list(self._tasks.items()):
                if gid < gen_id and not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                    self._tasks.pop(gid, None)

            # 3) mark adoption (mirror to _latest_gen for backward compatibility)
            self.adopted_gen = gen_id
            self._latest_gen = gen_id

            # (optional) TTFT log
            st = self._start_ts.get(gen_id)
            if st:
                ttft = time.time() - st
                log("info", f"LLM adopted (gen {gen_id}) TTFT={ttft:.3f}s")

            # 4) enforce the stream budget after cancellations
            await self._enforce_stream_limit()

    async def _enforce_stream_limit(self):
        """Keep the adopted gen and the newest not-yet-emitting ones. Cancel others."""
        if self.max_concurrent_streams < 1:
            return

        live = [(gid, t) for gid, t in self._tasks.items() if not t.done()]
        if len(live) <= self.max_concurrent_streams:
            return

        live_gids = [gid for gid, _ in live]
        keep: set[int] = set()

        # Always keep the adopted gen if it is alive
        if self.adopted_gen in live_gids:
            keep.add(self.adopted_gen)

        # Prefer newest "warm standbys" that have NOT emitted yet
        not_emitting_newest = [
            gid
            for gid, _ in sorted(live, key=lambda x: x[0], reverse=True)
            if self._first_emit_ts.get(gid, 0.0) == 0.0 and gid != self.adopted_gen
        ]

        budget = self.max_concurrent_streams - len(keep)
        if budget > 0:
            keep.update(not_emitting_newest[:budget])

        # Cancel the rest
        for gid, t in live:
            if gid not in keep and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                self._tasks.pop(gid, None)

    async def _stream_single(self, messages, gen_id: int):
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                # extract text
                if not (chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content):
                    continue
                text = (chunk.choices[0].delta.content or "").strip()
                if not text:
                    continue

                # mark first emission time if this is the first token for this gen
                if not self._first_emit_ts.get(gen_id, 0.0):
                    self._first_emit_ts[gen_id] = time.time()

                    # adopt on first token if newer than current adoption
                    if gen_id > self.adopted_gen:
                        await self._adopt_generation(gen_id)
                    else:
                        # older than the already adopted one => ignore this gen
                        continue

                # forward only if this gen is the adopted speaker
                if gen_id != self.adopted_gen:
                    continue

                try:
                    await self.server_state.llm_event_queue.put(("append", gen_id, text))
                except asyncio.QueueFull:
                    log("warning", "llm_event_queue full; dropping LLM chunk")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("error", f"LLM gen {gen_id} streaming error: {e}")
        finally:
            self._tasks.pop(gen_id, None)

    async def stop(self):
        # cancel all live tasks
        for gid, t in list(self._tasks.items()):
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

        # reset adoption bookkeeping
        self.adopted_gen = 0
        self._latest_gen = 0
        self._first_emit_ts.clear()
        self._start_ts.clear()


# -----------------------
# Server
# -----------------------
@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock
    asr_processor: AsyncASRProcessor | None = None

    def __init__(
        self,
        model_type: str,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        cfg_coef: float,
        device: str | torch.device,
        enable_asr: bool = True,
        min_restart_interval: float = 0.50,
        max_concurrent_streams: int = 7,
        **kwargs,
    ):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # LLM events from multiplexer
        self.llm_event_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        # ASR
        self.asr_processor = AsyncASRProcessor(sample_rate=int(self.mimi.sample_rate)) if enable_asr else None
        self._require_initialized_asr(enable_asr)

        # Multiplexer (parallel streams, single adopted stream)
        self.llm_mux = LLMStreamMultiplexer(
            server_state=self,
            system_prompt="""
You are Moshi, talking with the User. The User is currently mid-conversation.
Predict the flow of the User's dialogue and generate a suitable next response accordingly.
Generate only the dialogue directly, without any additional commentary.
Speak confidently on the predicted topic—no need to ask for confirmation.
Your answer must be short and concise, max 30 words. Do not prefix with moshi:.
Provide correct information consistent with the User's statements.
Since the output will be spoken, avoid symbols not needed for pronunciation (e.g., " ー ;).
""".strip(),
            min_restart_interval=min_restart_interval,
            max_prompt_chars=6000,
            max_concurrent_streams=max_concurrent_streams,
        )

        self.loop: asyncio.AbstractEventLoop | None = None

    def _require_initialized_asr(self, enable_asr: bool) -> None:
        if not enable_asr:
            return
        if self.asr_processor is not None and self.asr_processor.asr_enabled:
            return
        reason = "unknown error"
        if self.asr_processor is not None and self.asr_processor.init_error:
            reason = self.asr_processor.init_error
        raise RuntimeError(
            "ASR is enabled but Google Speech-to-Text could not be initialized. "
            f"{reason} "
            "Set GOOGLE_APPLICATION_CREDENTIALS to a valid Google Cloud service account credential file "
            "or rerun with --no-enable-asr."
        )

    # ASR -> LLM nudgers
    def _llm_nudge_partial(self, full_text: str):
        if self.loop:
            self.llm_mux.on_interim_pending(full_text)

    def _llm_nudge_force(self):
        if self.loop:
            self.llm_mux.nudge_now(force=True)

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        resolved_device = torch.device(self.device)
        if torch.cuda.is_available() and resolved_device.type == "cuda":
            torch.cuda.synchronize(device=resolved_device)

    def __del__(self):
        if hasattr(self, "asr_processor") and self.asr_processor:
            self.asr_processor.running = False

    async def handle_chat(self, request):
        if self.lock.locked():
            return web.Response(status=503, text="Server busy - another session is active")

        # reset conversation
        global conversation_text, current_speaker

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        close = False

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    data = message.data
                    if not isinstance(data, bytes):
                        log("error", f"unsupported message type {type(data)}")
                        continue
                    if len(data) == 0:
                        log("warning", "empty message")
                        continue
                    kind = data[0]
                    if kind == 1:  # audio
                        payload = data[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        async def opus_loop():
            """Single owner of lm_gen. Drains LLM events and synthesizes audio."""
            all_pcm_data = None
            skip_frames = 1
            active_gen = 0  # latest generation id

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                # 1) apply latest LLM text stream (only newest gen)
                try:
                    while True:
                        etype, gen_id, text = self.llm_event_queue.get_nowait()
                        if etype == "append":
                            if gen_id != active_gen:
                                self.lm_gen.update_oracle_tokens_streaming(None, reset=True)
                                active_gen = gen_id
                                ts = int(time.time() * 1000)
                                append_session_log("oracle_stream.txt", f"{ts}: [RESET]\n")
                                append_session_log("llm_stream_words.txt", f"{ts}: 32000\n")
                            token_ids = list(self.text_tokenizer.encode(text))  # type: ignore[attr-defined]
                            self.lm_gen.update_oracle_tokens_streaming(token_ids, reset=False)
                            log("info", f"[LLM] {text}")
                            ts = int(time.time() * 1000)
                            append_session_log("oracle_stream.txt", f"{ts}: {text}\n")
                            append_session_log("llm_stream_words.txt", f"{ts}: {text}\n")
                except asyncio.QueueEmpty:
                    pass

                # 2) audio pipeline
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]

                    # feed ASR with the incoming mic chunk
                    if self.asr_processor:
                        self.asr_processor.process_audio(chunk.copy())

                    # run moshi encode/decode with oracle tokens from LLM
                    chunk_t = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk_t)
                    if skip_frames:
                        self.mimi.reset_streaming()
                        skip_frames -= 1
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        # surface any generated text tokens
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore[attr-defined]
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            add_to_conversation("moshi", _text.strip(), flush_file=False)
                            ts = int(time.time() * 1000)
                            append_session_log("moshi_words.txt", f"{ts}: {_text.strip()}\n")
                            await ws.send_bytes(msg)

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        log("info", "accepted connection")
        async with self.lock:
            with conversation_lock:
                conversation_text = ""
                current_speaker = None
            clear_session_logs()
            append_session_log("conversation.txt", "")

            await self.llm_mux.stop()
            try:
                while True:
                    self.llm_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            self.loop = asyncio.get_running_loop()
            self.llm_mux.set_loop(self.loop)

            if self.asr_processor:
                self.asr_processor.set_llm_nudgers(self._llm_nudge_partial, self._llm_nudge_force)
                await self.asr_processor.start()

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            await ws.send_bytes(b"\x00")  # handshake
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())

        # Cleanup
        await self.llm_mux.stop()
        if self.asr_processor:
            await self.asr_processor.stop()
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true", help="Activate a gradio tunnel.")
    parser.add_argument(
        "--gradio-tunnel-token", help="Provide a custom (secret) token here to keep getting the same URL."
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo", type=str, default=loaders.DEFAULT_REPO, help="HF repo to look into, defaults Moshiko."
    )
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument(
        "--no_fuse_lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
        help="Do not fuse LoRA layers into Linear layers.",
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16.",
    )
    parser.add_argument(
        "--enable-asr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ASR processing for transcription (default: True)",
    )
    parser.add_argument(
        "--min-restart-interval",
        type=float,
        default=0.50,
        help="Minimum interval in seconds between parallel LLM stream starts.",
    )
    parser.add_argument(
        "--max-concurrent-streams",
        type=int,
        default=5,
        help="Maximum number of concurrent parallel LLM streams.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for plaintext local session logs. If omitted, no local "
            "conversation or token logs are written. Can also be set via MOSHI_LOG_DIR."
        ),
    )
    args = parser.parse_args()
    seed_all(42424242)
    configure_save_dir(args.log_dir or os.environ.get("MOSHI_LOG_DIR"))

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log(
                "error",
                "Cannot find gradio which is required to activate a tunnel. Please install with pip install gradio.",
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token or secrets.token_urlsafe(32)

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        lora_weights=args.lora_weight,
        config_path=args.config_path,
    )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    log("info", "moshi loaded")

    state = ServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        args.cfg_coef,
        args.device,
        enable_asr=args.enable_asr,
        min_restart_interval=args.min_restart_interval,
        max_concurrent_streams=args.max_concurrent_streams,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "warming up the model")
    state.warmup()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                extract_data_archive(tar, dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        static_path = args.static

    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if args.enable_asr:
        log("info", "ASR enabled – partials commit immediately; single active LLM stream restarts during speech")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None, **tunnel_kwargs)
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


def cli():
    with torch.no_grad():
        main()


if __name__ == "__main__":
    cli()
