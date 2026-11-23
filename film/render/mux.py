# -*- coding: utf-8 -*-
"""把配乐混进成片 —— 视频流直接 copy，只重编音频。

用法::

    python film/render/mux.py                          # v1 + score.wav -> v1_scored.mp4
    python film/render/mux.py --video out/x.mp4 --audio out/score.wav
    python film/render/mux.py --gain-db -2            # 想再压低一点

设计要点
--------
* **视频不重编**（`-c:v copy`）：706 秒的片子重编一次要好几分钟，而画面已经
  是最终码率了，没有必要；
* **`-shortest`**：音频比画面长/短都裁到较短者，避免尾部出现静音黑帧；
* **响度检查**：混完用 `volumedetect` 读平均/峰值音量，写进输出，避免"以为
  混上了其实增益是 -inf"。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(HERE)


def ffmpeg_exe():
    if os.environ.get('WORLD3D_FFMPEG'):
        return os.environ['WORLD3D_FFMPEG']
    w = shutil.which('ffmpeg')
    if w:
        return w
    raise SystemExit('找不到可用的 ffmpeg')


def duration_of(path):
    r = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', path],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', r.stderr or '')
    if not m:
        raise RuntimeError(f'读不到时长: {path}')
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def volume_of(path):
    """返回 (mean_db, max_db)；解析不到返回 (None, None)。"""
    r = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', path,
                        '-af', 'volumedetect', '-f', 'null', '-'],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    txt = r.stderr or ''
    mean = re.search(r'mean_volume:\s*(-?[\d.]+) dB', txt)
    peak = re.search(r'max_volume:\s*(-?[\d.]+) dB', txt)
    return (float(mean.group(1)) if mean else None,
            float(peak.group(1)) if peak else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=os.path.join(FILM_DIR, 'out',
                                                    'worldwright_v1.mp4'))
    ap.add_argument('--audio', default=os.path.join(FILM_DIR, 'out', 'score.wav'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--gain-db', type=float, default=0.0)
    ap.add_argument('--bitrate', default='192k')
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        raise SystemExit(f'视频不存在: {args.video}')
    if not os.path.exists(args.audio):
        raise SystemExit(f'音频不存在: {args.audio}（先跑 film/audio/score.py）')

    out = args.out or os.path.join(FILM_DIR, 'out', 'worldwright_v1_scored.mp4')
    vd, ad = duration_of(args.video), duration_of(args.audio)
    print(f'视频 {vd:.2f}s | 音频 {ad:.2f}s | 差 {ad - vd:+.2f}s')

    cmd = [ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
           '-i', args.video, '-i', args.audio,
           '-map', '0:v:0', '-map', '1:a:0',
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', args.bitrate,
           '-shortest', '-movflags', '+faststart']
    if abs(args.gain_db) > 1e-6:
        cmd += ['-af', f'volume={args.gain_db}dB']
    cmd += [out]
    subprocess.run(cmd, check=True)

    mean, peak = volume_of(out)
    mb = os.path.getsize(out) / 1048576
    got = duration_of(out)
    print(f'[ok] {out}  {got:.2f}s  {mb:.1f}MB')
    print(f'     音量 mean={mean} dB / peak={peak} dB')
    if mean is None or mean < -50.0:
        print('[warn] 平均音量异常低，可能没混上')
        print('MUX_FAILED')
        return 1
    if abs(got - min(vd, ad)) > 0.15:
        print(f'[warn] 成片时长 {got:.2f} 与预期 {min(vd, ad):.2f} 差得多')
        print('MUX_FAILED')
        return 1
    print('MUX_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
