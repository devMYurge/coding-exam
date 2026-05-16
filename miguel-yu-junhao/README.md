# Miguel Yu Junhao — AI 编程面试提交

两道题目均已完成。本文件为提交说明 + 技术方案；统一的启动命令见 [`code/SETUP.md`](./code/SETUP.md)。

---

## 题目一：On-Call 助手

**位置**：[`code/question-1/`](./code/question-1/)
**入口**：[`main.py`](./code/question-1/main.py)（单文件实现）

### 技术栈

- Python 3.11
- FastAPI / Uvicorn — HTTP 服务
- BeautifulSoup4 — SOP HTML 正文抽取
- 自实现 TF-IDF（Phase 1）
- fastembed — 文档 / 查询向量化（Phase 2）
- LangChain + OpenAI `gpt-4o-mini` — Agent（Phase 3）

### 实现思路

按路由前缀切分三个阶段，全部集中在 `main.py`：

- **`/v1` — 关键词检索**。启动时从 `data/` 加载所有 SOP，BS4 抽出正文，自实现 TF-IDF（IDF 用 `log((N+1)/(df+1)) + 1`，归一化后余弦相似度），返回标题 + 命中片段。
- **`/v2` — 语义检索**。复用同一文档集合，启动时用 fastembed 一次性算出所有文档向量并缓存；查询时只算查询向量并取余弦相似度。
- **`/v3` — Agent 对话**。把 `readFile(doc_id)` 作为工具暴露给 `gpt-4o-mini`，让模型自主决定要读哪些 SOP。系统提示约束模型只回答 SOP 范围内的内容，并要求引用具体文档。

每个阶段同时提供 JSON API（便于自动化验证）与同前缀下的极简 HTML 页面（输入框 + 结果列表）。

### 运行

```bash
cd code/question-1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='sk-...'    # 仅 Phase 3 需要
python main.py                    # http://127.0.0.1:8001
```

---

## 题目二：Antigravity 动画复刻

**位置**：[`code/question-2/`](./code/question-2/)
**入口**：[`src/App.jsx`](./code/question-2/src/App.jsx) → [`src/Medusae.jsx`](./code/question-2/src/Medusae.jsx)

### 技术栈

- Vite + React 19
- `@react-three/fiber` 9（R3F）
- three.js 0.180
- 自定义 GLSL 顶点 / 片元着色器
- 粒子引擎：`@vibe-rational/medusae`（开源包，源自 `BreathDearMedusae` 仓库；本项目 `code/question-2/src/` 直接采用其源码以便阅读、修改与替换）

### 实现思路

参考视频 `landing.mov` 经多轮逐帧分析后定性为：白底 + 彩色短线粒子 + 跟随光标的柔性力场 + 低频呼吸感。我先用 Canvas2D 自实现了一版（pill confetti + 流场 + 深度分层 + 弹簧鼠标），但在与视频继续对比时发现：Antigravity 的"呼吸 + halo + rim"特征本质上是 GLSL 着色器层面的事；与其在 2D 上反复逼近，不如直接采用一套已经为同一参考站设计的开源 WebGL 引擎（Medusae），把它的源码拉进本仓库的 `src/` 完整保留，外层只写最小的 Vite + React 集成。

最终结构非常薄：

- **`src/Medusae.jsx`** — 粒子引擎本体。GLSL 顶点着色器里实现 jellyfish halo（`distFromMouse`、`breathCycle = sin(uTime * 0.8)`、`shapeFactor` 噪声扰动、rim 影响推动粒子）、外圈震荡、按速度的拉伸/压缩、按方向的旋转；片元着色器是圆角 blob 的 alpha mask 与三色渐变（`uParticleColorOne/Two/Three`）按时间 + 位置混合。JS 层是 100 × 55 的 instanced 网格 + `useFrame` 中 `uMouse` 的 lerp 跟随（`dragFactor`）。
- **`src/defaults.js`** — 全部可调旋钮：cursor 半径 / 强度 / 拖拽系数、halo 振幅 / 频率 / 边缘宽度 / 缩放、粒子基础 / 激活尺寸、blob scale、旋转速度 / 抖动、三主色。
- **`src/App.jsx`** — 唯一的覆盖：`config={{ background: { color: '#ffffff' } }}` 以匹配 Antigravity 的浅色主题；其余全部沿用 defaults（默认值本身就是为 Antigravity 调过的）。
- **外层（`index.html` / `main.jsx` / `vite.config.js` / `package.json` / `index.css`）** — 本项目所写的最小 Vite + React 脚手架，目的是让 `Medusae.jsx` 跑起来并支持热重载。

### 设计取舍

- **早期路径**：纯 Canvas2D 自实现（pill 形粒子 + Poisson 分布 + 流场 + 三层景深 + 弹簧光标 + 点击脉冲环），过程中曾经历过 Three.js + 实例化变形的中间版本。这些迭代过程在 `prompt/` 的提示词截图里可以追溯。
- **最终决策**：在视觉与节奏层面，参考视频的"光学呼吸感"难以纯 2D 复刻；切换到 Medusae 后视觉一致性显著提升，同时保留了源码可读、可改、可被替换的属性（不是 `import` 黑盒）。
- **代码精简**：删除了所有早期的 hero 文案、控制面板、CTA 按钮等装饰层，最终页面只承载粒子背景本身。

### 运行

```bash
cd code/question-2
npm install
npm run dev    # http://localhost:5173
```

---

## AI 工具使用

整个开发过程主要借助 Claude 进行：

- 题目一：FastAPI 三阶段实现、TF-IDF 公式选择、fastembed 接入、LangChain Agent 的工具设计与系统提示打磨。
- 题目二：参考视频的多轮逐帧分析、Canvas2D 自实现的若干版迭代、Medusae 切换决策、Vite 脚手架搭建、目录清理。

所有提示词截图按顺序放在 [`prompt/`](./prompt/)，最终运行效果截图放在 [`screenshot/`](./screenshot/)。

---

## 目录结构

```
miguel-yu-junhao/
├── README.md                ← 本文件（技术方案 + 实现思路）
├── code/
│   ├── SETUP.md             ← 两道题统一启动说明
│   ├── question-1/          ← FastAPI On-Call 助手
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── data/            ← 10 份 SOP HTML
│   │   └── README.md        ← 题目一原题
│   └── question-2/          ← Vite + React + Medusae
│       ├── src/
│       │   ├── Medusae.jsx  ← 粒子引擎（GLSL + R3F）
│       │   ├── defaults.js
│       │   ├── medusae.css
│       │   ├── App.jsx
│       │   ├── main.jsx
│       │   └── index.css
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── README.md        ← 题目二原题
├── prompt/                  ← 提示词截图（按 01-xxx.png 编号）
├── screenshot/              ← 效果截图（按 01-overview.png 编号）
└── miguel-yu-junhao.pdf     ← 个人简历（待补）
```

---

## 打包与提交

```bash
# 从 miguel-yu-junhao 的上一层目录执行
zip -r miguel-yu-junhao-exam.zip miguel-yu-junhao \
    -x "*/node_modules/*" \
       "*/.venv/*" \
       "*/__pycache__/*" \
       "*/dist/*" \
       "*/.DS_Store"
```

排除 `node_modules/` / `.venv/` / `__pycache__/` / `dist/` 以控制压缩包体积；`.git/` 目录保留以满足"完整 Git 提交历史"的要求。
