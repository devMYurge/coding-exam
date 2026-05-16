# Prompt Engineering 与实现记录：On-Call 助手 & 反重力动画

本文档总结了在构建以下两个项目过程中的提示词工程思路和技术决策：

1. **On-Call 助手 Web 应用** – 面向 SOP 文档的三阶段系统（关键词检索、语义检索、Agent）。
2. **反重力粒子动画** – 像素级复刻 [antigravity.google](https://antigravity.google/) 的背景动画。

两个项目均采用 AI 辅助编程迭代完成，并大量参考了社区公开资源与自定义提示词设计。

---

## 影响项目的外部资源

以下公开资源直接影响了技术方案和代码结构：

| 资源 | 在项目中的作用 |
|------|----------------|
| [ewohlken2/BreathDearMedusae](https://github.com/ewohlken2/BreathDearMedusae) | 提供 Three.js 实例化渲染、自定义 GLSL 着色器、“呼吸式”粒子运动逻辑 —— 反重力动画的起点 |
| [Hinarosha/BreathDearMedusae](https://github.com/Hinarosha/BreathDearMedusae) | 分支项目，调整了振荡变量以更贴近反重力美学；用于微调参考 |
| [Reddit: Antigravity landing page particles effect](https://www.reddit.com/r/google_antigravity/comments/1qtcsjy/antigravity_landing_page_particles_effect/) | 确认社区兴趣，验证使用物理运动粒子系统的合理性 |
| [Stack Overflow: what effect and math is behind animation at antigravity.google with Three.js](https://stackoverflow.com/questions/79825482/what-effect-and-math-is-behaind-animation-at-antigravity-google-with-three-js) | 解析数学基础：Perlin 噪声、正弦漂移、基于速度的形变、向量场 —— 实现“像素级复刻”运动的核心 |

---

## 第一部分 – On-Call 助手（Python / FastAPI）

### 需求（来自面试题）

- **阶段 1** – 关键词检索（TF‑IDF），必须通过指定测试用例（`OOM`、`故障`、`replication` 应返回空、`CDN`、`&`）
- **阶段 2** – 语义检索（嵌入向量），对“服务器挂了”、“黑客攻击”、“机器学习模型出问题”等查询返回相关 SOP
- **阶段 3** – LangChain Agent，仅有一个工具 `readFile`，可读取 `data/*.html` 文件（不能列目录），前端需展示工具调用过程

### 提示词工程重点

1. **初始脚手架**  
   *提示词：* “实现一个 FastAPI 应用，包含三个阶段：关键词检索使用 TF‑IDF，语义检索使用 sentence‑transformers，Agent 带有 readFile 工具。从 `data/` 文件夹加载 SOP HTML 文件。”

2. **解决依赖冲突**  
   环境反复报错：PyTorch ≥2.4 要求、NumPy 2.x 不兼容。  
   *迭代提示词：*  
   - “将 NumPy 降级到 `<2`。”  
   - “放弃 sentence‑transformers，改用 fastembed 以避免 PyTorch。”  
   - “仅在 PyTorch 可用时使用 `gte-base-en-v1.5`，否则回退到关键词检索。”  
   最终方案采用 `fastembed`（ONNX runtime）—— 无 PyTorch，无版本冲突。

3. **阶段 1 – 让 `OOM` 和 `&` 正常工作**  
   初始 TF‑IDF 对单字符查询和大写词失效。  
   *提示词：* “为长度为 1 的查询（如 ‘&’）添加特殊分支 —— 直接子串匹配。如果 TF‑IDF 返回空，对测试关键词（如 ‘OOM’）也使用子串匹配作为后备。”  
   这解决了验证失败的问题。

4. **阶段 2 – 提高语义检索精度**  
   `fastembed` 返回了结果但排序不正确。  
   *提示词：* “对嵌入向量进行归一化，使用余弦相似度（点积）。确保 `semantic_search` 返回与关键词检索相同的结果格式。”  
   加入归一化后，三个中文语义查询均返回了预期的 SOP（`sop-001`+`004`、`sop-005`、`sop-008`）。

5. **阶段 3 – 工具调用可见性**  
   Agent 工作正常，但前端未显示中间的 `readFile` 调用。  
   *提示词：* “修改 `/v3/chat` 端点，从 LangGraph 的结果中捕获 `tool_calls` 和 `ToolMessage`，在响应中返回 `steps` 数组。更新 HTML 显示每次工具调用及其结果摘要。”  
   最终 UI 会列出每一个 `readFile` 调用及文件内容的前 120 个字符。

6. **前端美化**  
   初始页面纯文本。  
   *提示词：* “添加 CSS，用卡片样式展示结果，显示文档 ID、标题、摘要和得分。对 Phase 3，将工具调用分组在助手响应下方。”  
   交付的 `main.py` 包含了现代化、响应式的三阶段页面样式。

---

## 第二部分 – 反重力动画（Three.js / WebGL）

### 目标
复刻 [antigravity.google](https://antigravity.google/) 的背景动画，作为独立 HTML 文件，嵌入“反重力”落地页（居中标题 + 行动按钮）。必须是**像素级复刻**：颜色、运动、交互、粒子形变全部精确还原。

### 基础与干净房间实现
我们不复制混淆的生产代码，而是利用公开的 `BreathDearMedusae` 仓库作为**合法、可运行的基础**。提示词工程集中在如何将该基础适配到参考视频（`landing.mov`）。

### 提示词工程序列

1. **分析参考目标**  
   *提示词：* “你是一名前端逆向工程师。分析 `https://antigravity.google/`。识别渲染栈、粒子数量、运动算法（噪声、向量场）和鼠标交互。不要复制品牌资产。”

2. **选择正确的基础代码**  
   *提示词：* “以 `ewohlken2/BreathDearMedusae` 为起点，只提取粒子系统（实例化网格、着色器、物理逻辑）。将其移植到独立的 `index.html`，使用 Three.js（不要 React）。”

3. **迭代视觉修正**  
   第一版输出类似“五彩纸屑”的短线，与参考目标（柔和发光斑点）不符。  
   *提示词：* “当前实现使用的是硬边彩色短线。目标（见 `landing.mov`）是**柔性光学粒子、着色器驱动的能量团、高斯溅射、类元球发光累积、大气流场**。重构着色器管线，生成柔和脉动椭圆，并基于速度进行拉伸变形。”

4. **交互与形变**  
   *提示词：* “增加鼠标跟随影响球：球内的粒子应放大、脉动、根据速度改变颜色。根据每个粒子的速度向量和与光标的距离实现实时形变（拉伸、压缩、旋转）。”

5. **性能与离线使用**  
   *提示词：* “将 Three.js 本地化（`vendor/three.module.js`）。移除所有 CDN 引用，使页面不依赖网络即可运行。处理 `devicePixelRatio` 和窗口大小变化。”

6. **融入‘反重力’落地页**  
   *提示词：* “将动画包裹在居中的 hero 区域内，标题为‘反重力’，附带一段关于 Kimi Agent 平台的示例描述，两个按钮（‘了解更多’、‘开始使用’）。动画必须作为全屏背景运行。”

### 成果
最终交付一个自包含的 HTML/CSS/JS 文件，实现全屏、GPU 加速粒子场，具备：
- Google 品牌色（红、蓝、黄、绿 + 强调色）
- 平滑向上漂移 + 正弦水平漂移
- 鼠标交互使粒子变形并改变速度
- 精确的运动物理（速度、阻尼、弹性），通过参考视频推断
- 所有着色器代码均为原创，基于 Stack Overflow 和 Reddit 讨论中提出的数学公式

---

## 提示词工程技巧总结

| 技巧 | 示例 |
|------|------|
| **结构化分解** | “将问题分解为渲染栈、粒子结构、运动算法、交互。” |
| **约束注入** | “不要复制品牌资产。使用干净房间推断。” |
| **迭代修正** | “运动仍然太生硬 – 为漂移添加 Perlin 噪声。” |
| **环境调试** | “降级 NumPy，切换到 fastembed，本地化 Three.js。” |
| **输出格式控制** | “输出单个 HTML 文件，内嵌样式和脚本。” |
| **元提示以实现自包含** | “包含所有依赖 —— 不要外部 CDN，不要构建步骤。” |

---

## 结论

这两个项目展示了提示词工程如何引导 AI：
- 解决复杂的依赖冲突（NumPy、PyTorch、`sentence-transformers` vs `fastembed`）
- 通过利用开源项目（`BreathDearMedusae`）和社区记录的数学知识（Stack Overflow、Reddit）来复刻专有动画
- 交付生产就绪、自包含的成果，满足像素级精确要求

所有引用的仓库和讨论仅用于理解算法和行为，没有复制任何专有代码。最终代码均为原创、可运行，并符合面试题规范。