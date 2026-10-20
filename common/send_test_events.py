"""Send simulated gesture events to a running game over the real UDP interface.

Start the game first:   python -m game --input gesture
Then in another terminal:

    python -m common.send_test_events --interactive          # type l / r / u / d + Enter
    python -m common.send_test_events --script left,right,jump,roll --interval 1.0
    python -m common.send_test_events --random --rate 1.5 --duration 30
    python -m common.send_test_events --random --fake-pipeline-ms 120 --min-conf 0.4

--fake-pipeline-ms back-dates t_capture and fills in stage timestamps, so the
latency report can be exercised end to end before a real classifier exists.
"""

from __future__ import annotations

import argparse
import random
import time

from .events import DEFAULT_HOST, DEFAULT_PORT, EventSender, EventType, now


def send(sender: EventSender, kind: EventType, conf: float, pipeline_ms: float):
    if pipeline_ms > 0:
        t = now()
        p = pipeline_ms / 1000
        ev = sender.send(kind, conf, t_capture=t - p, t_landmarks=t - 0.6 * p, t_inference=t - 0.1 * p)
    else:
        ev = sender.send(kind, conf)
    print(f"sent {ev.event_type.value:5s} conf={ev.confidence:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m common.send_test_events", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--interactive", action="store_true")
    mode.add_argument("--script", help="comma-separated actions, e.g. left,left,jump")
    mode.add_argument("--random", action="store_true")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between scripted events")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--rate", type=float, default=1.0, help="random mode: events per second")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--min-conf", type=float, default=1.0, help="random confidences in [min-conf, 1]")
    p.add_argument("--fake-pipeline-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = p.parse_args(argv)

    sender = EventSender(a.host, a.port, source="test_sender")
    rng = random.Random(a.seed)
    conf = lambda: rng.uniform(a.min_conf, 1.0)  # noqa: E731

    if a.interactive:
        print("type l/r/u/d (or left/right/jump/roll) and press Enter; Ctrl-D to quit")
        try:
            while True:
                line = input("> ").strip()
                if not line:
                    continue
                try:
                    send(sender, EventType.parse(line), 1.0, a.fake_pipeline_ms)
                except ValueError:
                    print(f"unknown action {line!r}")
        except (EOFError, KeyboardInterrupt):
            pass
    elif a.script:
        actions = [EventType.parse(s) for s in a.script.split(",")]
        for _ in range(a.repeat):
            for kind in actions:
                send(sender, kind, conf(), a.fake_pipeline_ms)
                time.sleep(a.interval)
    else:
        end = now() + a.duration
        while now() < end:
            time.sleep(rng.expovariate(a.rate))
            send(sender, rng.choice(list(EventType)), conf(), a.fake_pipeline_ms)
    sender.close()


if __name__ == "__main__":
    main()
