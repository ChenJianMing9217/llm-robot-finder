"""房子的「設計圖」：從 Gazebo 的房子模型讀出牆和家具，畫在地圖座標上。

  python house_plan.py route                           → tools/explore_route.png（巡邏路線）
  python house_plan.py check ../maps/house_map_v2.yaml → 把 SLAM 地圖疊上真正的牆：
                                                          每個房間還有多少沒看過、哪些黑點離真正的牆太遠（可能是重影）、
                                                          哪些真正的牆在地圖上不是黑色（缺口：Nav2 可能以為可以抄近路）
  python house_plan.py patch ../maps/house_map_v2.yaml → 只把「真正有牆、地圖上是灰色」的格子補成牆，存成 house_map_v2_fixed
                                                          （SLAM 地圖有破洞，參考平面圖補牆，真實工作也常這樣做）；
                                                          另外把雷射掃不到的矮東西（杯子、咖啡桌底座）畫進地圖

只算離地 0.3 公尺以下的東西（門楣、窗台在半空中：雷射掃不到，機器人也撞不到）。
地圖座標 = 世界座標 + (2.0, 0.5)：機器人出生在世界 (-2.0, -0.5)，SLAM 把出生點當成地圖原點。
在 Windows 執行（要中文字型）：uv run --no-project --with matplotlib --with pyyaml python house_plan.py route
"""
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
HOUSE_SDF = "/opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_house/model.sdf"
WORLD = HERE.parent / "ros2_ws/src/home_sim/worlds/house_cup.world"
PLACES = HERE.parent / "ros2_ws/src/nav_tools/places_house.yaml"
SPAWN = (-2.0, -0.5)                                 # 世界座標；地圖原點
# 家具：中文名稱、大約半徑（公尺，只用來檢查「黑點附近有沒有東西」）
FURNITURE = {"Mailbox": ("信箱", 0.25), "bookshelf": ("書架", 0.5), "cabinet": ("櫃子", 0.35),
             "cafe_table": ("咖啡桌", 0.35), "first_2015_trash_can": ("垃圾桶", 0.25), "table": ("桌子", 0.9),
             "table_marble": ("大理石桌", 1.2), "Refrigerator": ("冰箱", 0.6), "Room_Essentials_Mug_White_Yellow": ("杯子", 0.1),
             "Sofa": ("沙發", 1.1), "Chair": ("椅子", 0.4)}
ADDED = {"Refrigerator", "Room_Essentials_Mug_White_Yellow", "Sofa", "Chair"}


def to_map(x, y):
    return x - SPAWN[0], y - SPAWN[1]


def _pose(el):
    p = el.find("pose")
    v = [float(t) for t in p.text.split()] if p is not None and p.text else []
    return v + [0.0] * (6 - len(v))


def _read_sdf():
    if os.path.exists(HOUSE_SDF):                    # 在 WSL 裡直接讀
        return Path(HOUSE_SDF).read_text(encoding="utf-8")
    return subprocess.run(["wsl.exe", "-e", "cat", HOUSE_SDF], capture_output=True, check=True).stdout.decode("utf-8")


def load():
    """回傳 (牆, 家具)。牆：[(地圖中心 x, y, 方向弧度, 長, 寬)]；家具：[(中文名, 地圖 x, y, 半徑, 是不是新加的)]。"""
    house = ET.fromstring(_read_sdf()).find("model")
    walls = []
    for link in house.findall("link"):
        lx, ly, lz, _, _, lyaw = _pose(link)
        for col in link.findall("collision"):
            size = col.find("geometry/box/size")
            if size is None:
                continue
            cx, cy, cz, _, _, cyaw = _pose(col)
            sx, sy, sz = map(float, size.text.split())
            if lz + cz - sz / 2 >= 0.3:              # 門楣、窗台：在半空中
                continue
            wx = lx + cx * math.cos(lyaw) - cy * math.sin(lyaw)
            wy = ly + cx * math.sin(lyaw) + cy * math.cos(lyaw)
            walls.append((*to_map(wx, wy), lyaw + cyaw, sx, sy))
    things = []
    for m in house.findall("model"):
        base = next((k for k in sorted(FURNITURE, key=len, reverse=True) if m.get("name").startswith(k)), None)  # 長的先比：table_marble 不是 table
        if base:
            x, y = _pose(m)[:2]
            things.append((FURNITURE[base][0], *to_map(x, y), FURNITURE[base][1], False))
    for inc in ET.parse(WORLD).getroot().iter("include"):
        uri = inc.findtext("uri") or ""
        name = uri.rstrip("/").split("/")[-1]
        if name in ADDED:
            x, y = _pose(inc)[:2]
            things.append((FURNITURE[name][0], *to_map(x, y), FURNITURE[name][1], True))
    return walls, things


def _font():
    names = {x.name for x in font_manager.fontManager.ttflist}
    for f in ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK SC"]:
        if f in names:              # Ubuntu：sudo apt install fonts-noto-cjk
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


def _corners(w):
    cx, cy, yaw, sx, sy = w
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + u * c - v * s, cy + u * s + v * c) for u, v in
            [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]]


def draw_plan(ax, walls, things, rooms, wall_color="#333", fill=True):
    for w in walls:
        ax.add_patch(plt.Polygon(_corners(w), closed=True, fill=fill, fc=wall_color, ec=wall_color, lw=1))
    for name, x, y, _, added in things:
        ax.plot(x, y, "s" if added else "o", color="tab:red" if added else "gray", ms=6)
        ax.text(x + 0.12, y - 0.05, name, fontsize=8, color="tab:red" if added else "dimgray")
    for name, r in rooms.items():
        ax.text(sum(r["x"]) / 2, sum(r["y"]) / 2, name, fontsize=15, color="tab:blue", alpha=0.35, ha="center", va="center")


def route_figure(out):
    walls, things = load()
    data = yaml.safe_load(PLACES.read_text(encoding="utf-8"))
    route = data["explore"]
    _font()
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=90)
    draw_plan(ax, walls, things, data["rooms"])
    for i in range(len(route) - 1):                  # 每一段一個箭頭；回程顏色淡一點
        a, b = route[i], route[i + 1]
        back = any((r["x"], r["y"]) == (b["x"], b["y"]) for r in route[:i + 1])
        ax.annotate("", xy=(b["x"], b["y"]), xytext=(a["x"], a["y"]),
                    arrowprops=dict(arrowstyle="-|>", color="#9ad1a6" if back else "#1f9d55", lw=1.6 if back else 2.2,
                                    shrinkA=9, shrinkB=9))
    spots = {}                                       # 同一個位置可能經過好幾次：編號寫在一起
    for i, r in enumerate(route, 1):
        spots.setdefault((r["x"], r["y"]), []).append(i)
    for (x, y), nums in spots.items():
        ax.plot(x, y, "o", ms=17 if len(nums) == 1 else 21, color="#1f9d55", mec="white", mew=1.5, zorder=5)
        ax.text(x, y, ",".join(map(str, nums)) if len(nums) < 3 else f"{nums[0]},…", color="white", fontsize=8 if len(nums) > 1 else 9,
                ha="center", va="center", fontweight="bold", zorder=6)
    ax.plot(0, 0, "*", ms=16, color="gold", mec="k", zorder=7)
    ax.set_title("巡邏路線（畫地圖用）：ros2 run nav_tools explore --places house\n"
                 "照編號開（深綠 = 第一次經過、淺綠 = 回頭）；黑 = 牆、灰點 = 房子原本的家具、紅方塊 = 我們加的；★ = 起點（地圖原點）",
                 fontsize=11)
    ax.set_xlim(-6, 10.2)
    ax.set_ylim(-5.2, 6.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    print(f"存好了：{out}")


def _read_map(map_yaml):
    info = yaml.safe_load(Path(map_yaml).read_text(encoding="utf-8"))
    pgm = Path(map_yaml).parent / info["image"]
    with open(pgm, "rb") as fh:
        assert fh.readline().strip() == b"P5"
        line = fh.readline()
        while line.startswith(b"#"):
            line = fh.readline()
        w, h = map(int, line.split())
        fh.readline()
        img = np.frombuffer(fh.read(), dtype=np.uint8).reshape(h, w)
    return img, float(info["resolution"]), info["origin"][0], info["origin"][1]


def check_figure(map_yaml, out):
    walls, things = load()
    rooms = yaml.safe_load(PLACES.read_text(encoding="utf-8"))["rooms"]
    img, res, ox, oy = _read_map(map_yaml)
    h, w = img.shape
    # 每一格的中心座標（地圖座標）
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (h - np.arange(h) - 0.5) * res
    print(f"地圖 {w}×{h} 格（{w * res:.1f}×{h * res:.1f} 公尺），原點 ({ox:.2f}, {oy:.2f})")
    print("各房間還沒看過（灰色）的比例：")
    for name, r in rooms.items():
        cols = (xs >= r["x"][0]) & (xs <= r["x"][1])
        rows = (ys >= r["y"][0]) & (ys <= r["y"][1])
        sub = img[np.ix_(rows, cols)]
        area = (r["x"][1] - r["x"][0]) * (r["y"][1] - r["y"][0])
        seen = sub.size * res * res
        unknown = (np.sum(sub == 205) + max(area - seen, 0) / (res * res)) / (area / (res * res))
        print(f"  {name}：{unknown:.0%}")
    # 黑點離真正的牆、家具多遠：超過 0.3 公尺 → 可能是重影（打滑時畫歪的牆）
    r_idx, c_idx = np.nonzero(img == 0)
    px, py = xs[c_idx], ys[r_idx]
    dist = np.full(px.shape, np.inf)
    for cx, cy, yaw, sx, sy in walls:
        dx, dy = px - cx, py - cy
        u = np.abs(dx * math.cos(yaw) + dy * math.sin(yaw)) - sx / 2
        v = np.abs(-dx * math.sin(yaw) + dy * math.cos(yaw)) - sy / 2
        dist = np.minimum(dist, np.hypot(np.maximum(u, 0), np.maximum(v, 0)))
    for _, fx, fy, rad, _ in things:
        dist = np.minimum(dist, np.maximum(np.hypot(px - fx, py - fy) - rad, 0))
    stray = dist > 0.3
    print(f"黑點 {len(px)} 格，其中 {stray.sum()} 格離真正的牆和家具超過 30 公分（可能是重影）")
    # 真正的牆在地圖上畫到了嗎：沿著每面牆的中線每 5 公分看一格（牆有 15 公分厚，SLAM 的黑線畫在牆的表面 → 中線旁邊 2 格內有黑色就算畫到）
    gaps = []
    for cx, cy, yaw, sx, sy in walls:
        length = max(sx, sy)
        ux, uy = (math.cos(yaw), math.sin(yaw)) if sx >= sy else (-math.sin(yaw), math.cos(yaw))
        for t in np.arange(-length / 2 + 0.05, length / 2 - 0.05 + 1e-9, res):
            x, y = cx + t * ux, cy + t * uy
            c, r = int((x - ox) / res), int(h - (y - oy) / res)
            if not (2 <= c < w - 2 and 2 <= r < h - 2):
                continue                                 # 在地圖範圍外（沒去過的地方）
            near = img[r - 2:r + 3, c - 2:c + 3]
            if not (near == 0).any():
                gaps.append((x, y, "灰" if (near == 205).all() else "白"))
    gx = np.array([g[0] for g in gaps]); gy = np.array([g[1] for g in gaps])
    white = sum(g[2] == "白" for g in gaps)
    print(f"真正的牆在地圖上沒畫成黑色：{len(gaps)} 格（約 {len(gaps) * res:.1f} 公尺；其中「白色」{white} 格 = 地圖說可以走）")
    _font()
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=90)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[ox, ox + w * res, oy, oy + h * res])
    draw_plan(ax, walls, things, rooms, wall_color="tab:red", fill=False)
    ax.plot(px[stray], py[stray], "s", ms=2.5, color="orange", label="離真正的牆太遠（可能是重影）")
    if gaps:
        ax.plot(gx, gy, "x", ms=4, color="magenta", label="真正的牆沒畫成黑色（缺口）")
    ax.set_title(f"{Path(map_yaml).stem}：SLAM 地圖（白 = 看過、黑 = 牆、灰 = 沒看過）＋ 紅框 = 真正的牆", fontsize=12)
    ax.legend(loc="lower left")
    ax.set_xlim(-6, 10.2)
    ax.set_ylim(-5.2, 6.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out)
    print(f"存好了：{out}")


LOW = {"杯子": ("disc", 0.10), "咖啡桌": ("square", 0.56)}   # 雷射掃不到的矮東西：杯子約 10 公分高、咖啡桌底座 4 公分高


def patch_map(map_yaml):
    walls, things = load()
    img, res, ox, oy = _read_map(map_yaml)
    img = img.copy()
    h, w = img.shape
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (h - np.arange(h) - 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros(img.shape, dtype=bool)
    for cx, cy, yaw, sx, sy in walls:                # 每一格的中心在不在某面牆的長方形裡
        dx, dy = gx - cx, gy - cy
        u = np.abs(dx * math.cos(yaw) + dy * math.sin(yaw))
        v = np.abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
        inside |= (u <= sx / 2) & (v <= sy / 2)
    fill = inside & (img == 205)
    img[fill] = 0
    # 雷射（離地約 12 公分）掃不到的矮東西：Nav2 不知道它們在那裡，會規劃出穿過去的路 → 機器人撞上推不動的杯子就卡死
    # （第一次正式測驗：跑了 1.5 小時，從第 7 題起全部卡在廚房杯子旁邊）→ 直接畫進地圖（白色也蓋掉）
    low = np.zeros(img.shape, dtype=bool)
    for name, x, y, _, _ in things:
        if name in LOW:
            shape, size = LOW[name]
            low |= (np.hypot(gx - x, gy - y) <= size) if shape == "disc" else \
                   (np.abs(gx - x) <= size / 2) & (np.abs(gy - y) <= size / 2)
    img[low] = 0
    src = Path(map_yaml)
    out = src.with_name(src.stem + "_fixed")
    with open(out.with_suffix(".pgm"), "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (w, h))
        fh.write(img.tobytes())
    info = src.read_text(encoding="utf-8").replace(f"image: {src.stem}.pgm", f"image: {out.stem}.pgm")
    out.with_suffix(".yaml").write_bytes(info.replace("\r\n", "\n").encode("utf-8"))
    print(f"補了 {fill.sum()} 格（真正有牆、原本是灰色）＋ {low.sum()} 格（雷射掃不到的矮東西）→ {out.with_suffix('.yaml')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "route"
    if cmd == "route":
        route_figure(HERE / "explore_route.png")
    elif cmd == "check":
        m = Path(sys.argv[2])
        check_figure(m, m.with_name(m.stem + "_check.png"))
    elif cmd == "patch":
        patch_map(Path(sys.argv[2]))
    else:
        print(__doc__)
