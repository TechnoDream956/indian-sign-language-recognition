"""
Unit Tests for Phase 5F: ISL AlphabetEngine & WordBuffer
=========================================================
Covers:
- WordBuffer operations: add_letter, add_space, backspace, clear_word, clear_sentence, commit_word
- AlphabetEngine state machine:
  1. 'A' stabilizing and emitting after 5 frames
  2. 'A' held continuously for many frames emitting exactly ONCE
  3. 'A' -> release (cooldown) -> 'A' emitting a second 'A'
  4. 'H' -> 'E' -> 'L' -> release -> 'L' -> 'O' forming "HELLO"
  5. HELLO + SPACE committing to "HELLO "
  6. BACKSPACE removing last character
  7. CLEAR clearing current word
  8. ENTER / commit_word committing word
  9. SPEAK invoking TTSEngine exactly once
  10. RESET clearing debounce state without corrupting sentence
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.alphabet_engine import WordBuffer, AlphabetEngine


class MockAlphabetModel:
    """Mock TensorFlow model returning predetermined class probabilities for testing."""

    def __init__(self, label_map):
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.current_letter = "A"
        self.confidence = 0.95

    def set_prediction(self, letter: str, confidence: float = 0.95):
        self.current_letter = letter
        self.confidence = confidence

    def __call__(self, inputs, training=False):
        probs = np.zeros((inputs.shape[0], len(self.label_map)), dtype=np.float32)
        idx = self.label_map[self.current_letter]
        probs[:, idx] = self.confidence
        # Distribute remaining prob
        rem = (1.0 - self.confidence) / max(1, len(self.label_map) - 1)
        for i in range(len(self.label_map)):
            if i != idx:
                probs[:, i] = rem

        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = probs
        return mock_tensor


class TestWordBuffer(unittest.TestCase):
    """Test suite for WordBuffer operations."""

    def setUp(self):
        self.buf = WordBuffer()

    def test_add_letter_and_current_word(self):
        self.buf.add_letter("h")
        self.buf.add_letter("E")
        self.buf.add_letter("l")
        self.buf.add_letter("L")
        self.buf.add_letter("o")
        self.assertEqual(self.buf.get_current_word(), "HELLO")
        self.assertEqual(self.buf.get_sentence(), "")

    def test_add_space_and_commit(self):
        self.buf.add_letter("H")
        self.buf.add_letter("I")
        self.buf.add_space()
        self.assertEqual(self.buf.get_current_word(), "")
        self.assertEqual(self.buf.get_sentence(), "HI ")
        self.assertEqual(self.buf.get_full_text(), "HI ")

    def test_backspace(self):
        self.buf.add_letter("C")
        self.buf.add_letter("A")
        self.buf.add_letter("T")
        self.assertEqual(self.buf.get_current_word(), "CAT")
        self.buf.backspace()
        self.assertEqual(self.buf.get_current_word(), "CA")
        self.buf.backspace()
        self.buf.backspace()
        self.assertEqual(self.buf.get_current_word(), "")

        # Backspacing when word is empty deletes from sentence
        self.buf.sentence = "HELLO "
        self.buf.backspace()
        self.assertEqual(self.buf.get_sentence(), "HELLO")

    def test_clear_word_and_sentence(self):
        self.buf.add_letter("T")
        self.buf.add_letter("E")
        self.buf.add_letter("S")
        self.buf.add_letter("T")
        self.buf.commit_word()
        self.buf.add_letter("N")
        self.buf.add_letter("E")
        self.buf.add_letter("W")
        self.assertEqual(self.buf.get_current_word(), "NEW")
        self.assertEqual(self.buf.get_sentence(), "TEST ")

        self.buf.clear_word()
        self.assertEqual(self.buf.get_current_word(), "")
        self.assertEqual(self.buf.get_sentence(), "TEST ")

        self.buf.clear_sentence()
        self.assertEqual(self.buf.get_sentence(), "")

    def test_clear_all(self):
        self.buf.add_letter("A")
        self.buf.commit_word()
        self.buf.add_letter("B")
        self.buf.clear_all()
        self.assertEqual(self.buf.get_current_word(), "")
        self.assertEqual(self.buf.get_sentence(), "")


class TestAlphabetEngineStateMachine(unittest.TestCase):
    """Test suite for AlphabetEngine temporal debounce, single emission, cooldown, and TTS."""

    def setUp(self):
        # Create standard 26-class label map
        self.label_map = {chr(ord('A') + i): i for i in range(26)}
        self.mock_model = MockAlphabetModel(self.label_map)
        self.mock_tts = MagicMock()
        self.mock_tts.speak.return_value = True

        # Initialize AlphabetEngine with mock model
        self.engine = AlphabetEngine(
            confidence_threshold=0.75,
            stability_frames=5,
            cooldown_seconds=0.8,
            tts_engine=self.mock_tts
        )
        self.engine.model = self.mock_model
        self.engine.label_map = self.label_map
        self.engine.inv_label_map = self.mock_model.inv_label_map
        self.dummy_features = np.zeros(63, dtype=np.float32)

    def test_a_emits_after_5_frames(self):
        """Test that signing 'A' emits 'A' into current word on exactly the 5th frame."""
        self.mock_model.set_prediction("A", 0.90)
        t = 1000.0

        for frame in range(1, 5):
            res = self.engine.process_features(self.dummy_features, current_time=t)
            self.assertEqual(res["raw_prediction"], "A")
            self.assertIsNone(res["stable_letter"])
            self.assertIsNone(res["emitted_letter"])
            self.assertEqual(self.engine.word_buffer.get_current_word(), "")
            t += 0.033

        # Frame 5: Should latch stable and emit 'A'
        res5 = self.engine.process_features(self.dummy_features, current_time=t)
        self.assertEqual(res5["raw_prediction"], "A")
        self.assertEqual(res5["stable_letter"], "A")
        self.assertEqual(res5["emitted_letter"], "A")
        self.assertEqual(self.engine.word_buffer.get_current_word(), "A")

    def test_a_held_many_frames_only_emits_once(self):
        """Holding 'A' continuously for 60 frames must emit exactly once."""
        self.mock_model.set_prediction("A", 0.95)
        t = 1000.0
        emissions = []

        for frame in range(60):
            res = self.engine.process_features(self.dummy_features, current_time=t)
            if res["emitted_letter"]:
                emissions.append(res["emitted_letter"])
            t += 0.033

        self.assertEqual(emissions, ["A"], "Holding 'A' should emit exactly once")
        self.assertEqual(self.engine.word_buffer.get_current_word(), "A")

    def test_a_release_a_emits_second_a(self):
        """A -> release (neutral/no-hand >= 0.8s) -> A emits second 'A'."""
        self.mock_model.set_prediction("A", 0.95)
        t = 1000.0

        # Emit first 'A' (5 frames)
        for _ in range(5):
            self.engine.process_features(self.dummy_features, current_time=t)
            t += 0.033
        self.assertEqual(self.engine.word_buffer.get_current_word(), "A")

        # Hand release for 1.0s (> 0.8s cooldown)
        for _ in range(15):
            self.engine.process_features(None, current_time=t)
            t += 0.033
        t += 0.5  # Ensure elapsed > 0.8s

        # Sign 'A' again for 5 frames
        for _ in range(5):
            self.engine.process_features(self.dummy_features, current_time=t)
            t += 0.033

        self.assertEqual(self.engine.word_buffer.get_current_word(), "AA")

    def test_hello_sequence(self):
        """Sign H -> E -> L -> release -> L -> O to produce 'HELLO'."""
        letters = ["H", "E", "L", "L", "O"]
        t = 1000.0

        for idx, letter in enumerate(letters):
            if idx > 0 and letter == letters[idx - 1]:
                # Repeated letter requires hand release & cooldown
                for _ in range(10):
                    self.engine.process_features(None, current_time=t)
                    t += 0.033
                t += 0.9  # Cooldown satisfied

            self.mock_model.set_prediction(letter, 0.92)
            # Run 5 frames to stabilize & emit
            for _ in range(5):
                self.engine.process_features(self.dummy_features, current_time=t)
                t += 0.033
            # Transition time
            t += 0.85

        self.assertEqual(self.engine.word_buffer.get_current_word(), "HELLO")

    def test_hello_plus_space(self):
        """Signing HELLO and adding space produces sentence 'HELLO '."""
        self.engine.word_buffer.current_word = "HELLO"
        self.engine.word_buffer.add_space()
        self.assertEqual(self.engine.word_buffer.get_sentence(), "HELLO ")
        self.assertEqual(self.engine.word_buffer.get_current_word(), "")

    def test_backspace_control(self):
        """Backspace removes last character from word."""
        self.engine.word_buffer.current_word = "HELLO"
        self.engine.word_buffer.backspace()
        self.assertEqual(self.engine.word_buffer.get_current_word(), "HELL")

    def test_clear_control(self):
        """Clear clears current word."""
        self.engine.word_buffer.current_word = "HELLO"
        self.engine.word_buffer.clear_word()
        self.assertEqual(self.engine.word_buffer.get_current_word(), "")

    def test_enter_commit(self):
        """Enter / commit_word commits word to sentence with trailing space."""
        self.engine.word_buffer.current_word = "HELLO"
        self.engine.word_buffer.commit_word()
        self.assertEqual(self.engine.word_buffer.get_sentence(), "HELLO ")
        self.assertEqual(self.engine.word_buffer.get_current_word(), "")

    def test_speak_calls_tts_once(self):
        """Speak calls TTS exactly once with sentence text."""
        self.engine.word_buffer.sentence = "HELLO WORLD "
        success = self.engine.speak_sentence(language="English")
        self.assertTrue(success)
        self.mock_tts.speak.assert_called_once_with("HELLO WORLD", "English")

    def test_reset_clears_debounce_without_corrupting_sentence(self):
        """Reset clears debounce/candidate state without corrupting sentence."""
        self.engine.word_buffer.sentence = "GOOD MORNING "
        self.engine.word_buffer.current_word = "AL"
        self.engine.consecutive_candidate = "P"
        self.engine.consecutive_count = 3
        self.engine.stable_letter = "P"

        self.engine.reset_debounce_state()
        self.assertIsNone(self.engine.consecutive_candidate)
        self.assertEqual(self.engine.consecutive_count, 0)
        self.assertIsNone(self.engine.stable_letter)
        # Sentence and word must be preserved
        self.assertEqual(self.engine.word_buffer.get_sentence(), "GOOD MORNING ")
        self.assertEqual(self.engine.word_buffer.get_current_word(), "AL")


if __name__ == "__main__":
    unittest.main()
