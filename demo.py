"""
ASD Screening Demo Script
==========================
Run this to test the gaze tracker and speech analyzer independently
without the full Tkinter GUI. Useful for testing on headless systems
or quick debugging.

Usage:
    python demo.py --mode gaze        # Run gaze tracking only
    python demo.py --mode speech      # Run speech analysis only
    python demo.py --mode full        # Run complete pipeline
"""

import sys
import os
import argparse
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_gaze(duration: int = 30):
    """Run gaze tracking demo with animated dot."""
    print(f"\n{'='*50}")
    print("GAZE TRACKING DEMO")
    print(f"{'='*50}")
    print("Instructions:")
    print("  - A blue dot will move across the screen")
    print("  - Follow the dot with your eyes")
    print(f"  - Test runs for {duration} seconds")
    print("  - Press 'q' to quit early\n")

    from modules.gaze_tracker import GazeTracker, DotAnimator

    W, H = 800, 600
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    tracker = GazeTracker(dot_follow_threshold=0.30)
    animator = DotAnimator(W, H)
    animator.reset()

    start_time = time.time()
    window = "ASD Gaze Demo - Press Q to quit"
    cv2.namedWindow(window)

    print("Starting in 2 seconds...")
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (W, H))

        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        # Get dot position and draw
        dot_pos = animator.get_dot_position()
        frame = animator.draw_dot(frame)

        # Process gaze
        frame, gaze_frame = tracker.process_frame(frame, dot_pos, timestamp=elapsed)

        # Status overlay
        remaining = int(duration - elapsed)
        total = len(tracker.frames)
        following = sum(1 for f in tracker.frames if f.following)
        ratio = following / total if total > 0 else 0

        cv2.putText(frame, f"Time: {remaining}s", (10, H-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Frames: {total}", (10, H-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Following: {ratio:.0%}", (10, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Compute results
    result = tracker.compute_session_results()
    print("\n" + "="*50)
    print("GAZE RESULTS")
    print("="*50)
    print(f"  Following Ratio:    {result.following_ratio:.1%}")
    print(f"  Avg Deviation:      {result.avg_gaze_deviation:.3f}")
    print(f"  Saccade Smoothness: {result.saccade_smoothness:.3f}")
    print(f"  Fixation Count:     {result.fixation_count}")
    print(f"  Mutual Gaze Ratio:  {result.mutual_gaze_ratio:.1%}")
    print(f"  Blink Rate:         {result.blink_rate:.0f}/min")
    print(f"  Frames Tracked:     {result.frames_tracked}")
    print(f"\n  ── ASD GAZE RISK SCORE: {result.asd_risk_score:.1f}/100 ──")
    return result


def demo_speech():
    """Run speech analysis demo."""
    print(f"\n{'='*50}")
    print("SPEECH ANALYSIS DEMO")
    print(f"{'='*50}")

    from modules.speech_analyzer import SpeechAnalyzer

    analyzer = SpeechAnalyzer()
    prompts = SpeechAnalyzer.SCREENING_PROMPTS

    print(f"Recording {len(prompts)} speech samples\n")

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt['instruction']}")
        input("Press ENTER to start recording...")
        print(f"Recording for {prompt['expected_duration']}s... SPEAK NOW")

        audio = analyzer.start_recording(duration=prompt['expected_duration'])

        features = analyzer.extract_features(audio)
        print(f"  Pitch std: {features.pitch_std:.1f} Hz")
        print(f"  Pause ratio: {features.pause_ratio:.1%}")
        print(f"  Speech rate: {features.speech_rate:.1f} syl/s")
        print(f"  Repetition: {features.repetition_score:.2f}")
        print(f"  Segment risk: {features.asd_risk_score:.1f}")

    result = analyzer.analyze_all_recordings()
    print("\n" + "="*50)
    print("SPEECH RESULTS")
    print("="*50)
    print(f"  Pitch Std Dev:     {result.features.pitch_std:.1f} Hz")
    print(f"  Pitch Range:       {result.features.pitch_range:.1f} Hz")
    print(f"  Pause Ratio:       {result.features.pause_ratio:.1%}")
    print(f"  Speech Rate:       {result.features.speech_rate:.1f} syl/s")
    print(f"  Rhythm Regularity: {result.features.rhythm_regularity:.2f}")
    print(f"  Repetition Score:  {result.features.repetition_score:.2f}")
    print(f"  Jitter:            {result.features.jitter:.3f}")
    if result.risk_factors:
        print(f"\n  Risk Factors:")
        for rf in result.risk_factors:
            print(f"    ⚠  {rf}")
    print(f"\n  ── ASD SPEECH RISK SCORE: {result.asd_risk_score:.1f}/100 ──")
    return result


def demo_full():
    """Run full pipeline demo."""
    print("Running FULL PIPELINE DEMO\n")

    gaze_result = demo_gaze(duration=20)
    speech_result = demo_speech()

    from models.mobilenet_asd import ASDFusionClassifier

    fusion = ASDFusionClassifier()
    gaze_risk = gaze_result.asd_risk_score if gaze_result else 50.0
    speech_risk = speech_result.asd_risk_score if speech_result else 50.0

    score, category, details = fusion.predict(gaze_risk, speech_risk)

    print("\n" + "="*50)
    print("FINAL FUSED RESULT")
    print("="*50)
    print(f"  Gaze Risk:    {gaze_risk:.1f}/100")
    print(f"  Speech Risk:  {speech_risk:.1f}/100")
    print(f"  Visual Risk:  {details['visual_risk']:.1f}/100")
    print(f"\n  ▸ FUSED SCORE: {score:.1f}/100")
    print(f"  ▸ CATEGORY:    {category}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASD Screening Demo")
    parser.add_argument("--mode", choices=["gaze", "speech", "full"],
                        default="full", help="Demo mode")
    parser.add_argument("--duration", type=int, default=30,
                        help="Gaze test duration in seconds")
    args = parser.parse_args()

    if args.mode == "gaze":
        demo_gaze(duration=args.duration)
    elif args.mode == "speech":
        demo_speech()
    else:
        demo_full()
