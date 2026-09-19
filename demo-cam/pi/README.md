# Pi-side configuration (wyhrp0)

Everything in this folder is deployed to the **Raspberry Pi**, not run on the
Mac. It exists because of two distinct messages in
`sudo journalctl -u mediamtx.service -f`:

```
mediamtx[949]: encoder_hardware_h264_encode(): ioctl(VIDIOC_QBUF) failed
mediamtx[792]: WAR [RTSP] [session 1497e783] reader is too slow, discarding 469 frames
```

Two different processes, two different problems.

## "reader is too slow"

`mediamtx` (PID 792) telling us that a **client** is not pulling frames off its
socket fast enough. Each reader has a fixed write queue; the publisher is never
blocked to wait for one, so when the queue fills the oldest frames are dropped.

469 frames inside three seconds is not a reader that is slightly behind — it is
a reader that stopped reading. Two sessions were doing it at once, which pointed
at a shared bottleneck rather than one bad client.

**The cause was on our side, and is fixed in the app:** `/video_feed` used to
call `gen_frames()` once per HTTP request, and each call opened its own RTSP
connection *and* its own YOLO inference loop. Every browser tab was a second
reader on the Pi and a second inference loop on the Mac competing for the same
CPU — and a loop busy doing inference is a loop not draining its socket. Closing
the tab did not reliably end the session either, so orphaned readers accumulated
across page reloads. `detector.py` now runs one capture thread for the whole
process and fans the finished frames out to every viewer.

Both demos have that restructure now: it landed in `demo-cam/v1/detector.py` and
was ported to `demo-cam/v2/detector.py`, which shares one capture thread the same
way, adds `VID_STRIDE` (the socket is still drained at full rate) and reports the
reader at `/stream/status` — `viewers`, `fps`, `reconnects`, `age_sec`.

`writeQueueSize` in `mediamtx.recommended.yml` is the remaining tuning: a bigger
shock absorber for brief hiccups. It cannot rescue a reader that is permanently
slower than real time.

## `ioctl(VIDIOC_QBUF) failed`

This one is from the `rpicamera` helper (PID 949), and it is what actually
**stops the stream**. The helper could not queue a buffer to the Pi's hardware
H.264 encoder; once it loops it does not recover, the camera path drops, and
every reader is torn down with it.

Both upstream issues about it were closed as *not planned*, so the approach is
avoid-and-recover:

- **Avoid** — lower resolution and frame rate, and keep the bitrate comfortably
  high. Starving the encoder is the documented trigger; people reproduce it at
  100 kbps and report stability from ~700 kbps up. `mediamtx.recommended.yml`
  sets 1280x720 @ 15 fps @ 2 Mbps.
- **Recover** — `mediamtx-watchdog.sh` restarts the service on a burst of these
  errors, so a wedged encoder costs a couple of seconds instead of the rest of
  the session.

## Install

```bash
# 1. Merge the keys you want into the live config, then restart.
sudoedit /usr/local/etc/mediamtx.yml          # compare with mediamtx.recommended.yml
sudo systemctl restart mediamtx.service

# 2. Install the encoder watchdog.
sudo install -m 755 mediamtx-watchdog.sh /usr/local/bin/mediamtx-watchdog
sudo install -m 644 mediamtx-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx-watchdog.service
```

## Check the Pi itself first

An undervolted or throttled Pi produces exactly this pattern, and no config
change will help until it is fixed:

```bash
vcgencmd get_throttled     # 0x0 is healthy; anything else is power or heat
vcgencmd measure_temp
grep -i cma /proc/meminfo  # CMA exhaustion also starves the encoder
```

## Confirming the fix

With `logLevel: info`, MediaMTX logs each reader connecting. Open the monitor
page on the Mac in three tabs: you should see **one** RTSP session appear, not
three. On the Mac side, `http://localhost:5005/stream/status` reports the shared
reader's fps, viewer count and reconnect count — a reconnect counter that climbs
while nobody is opening or closing tabs means the Pi is dropping the path, and
this journal is where to look next.
