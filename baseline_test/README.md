# baseline_test

`baseline_test` 是当前仓库中用于运行黑盒污染检测 baseline 的主目录。它面向 few-shot contamination 场景，统一在同一套评测入口下运行并输出多个 baseline 的结果，当前重点脚本是 [common.py](/d:/Desktop/CAT/Data%20Contamination/llm_eva/CDD-TED4LLMs/baseline_test/common.py)。

## 项目作用

本目录的目标是：

- 统一运行以下黑盒污染检测 baseline：
  - `Perplexity`
  - `Min-k% Prob`
  - `Reference-based`
  - `TS-Guessing`
  - `DE-COP`
  - `DCQ`
- 尽量复用 `lm-evaluation-harness` 的 leaderboard task 配置，而不是为每个数据集单独手写任务 prompt。
- 在 few-shot contamination 设定下，把“任务原始 prompt”和“检测型 prompt”分开管理。

## 当前设计原则

- 本目录处理的数据通常来自 `lm-evaluation-harness` 的测评记录，因此“任务本身怎么出题、怎么给标准答案、怎么给 few-shot 样例”应优先回到 harness task 配置中恢复。
- 任务本身的题目渲染、答案提取、choices 提取、few-shot 样例，优先来自 `lm-evaluation-harness`。
- 检测 baseline 自己的 probing prompt 由 `common.py` 内部实现。
- `Perplexity / Min-k / Reference-based` 主要依赖任务 prompt，因此对 harness 记录型数据集通常更容易自适应。
- `TS-Guessing / DE-COP / DCQ` 除了任务 prompt，还依赖额外的检测 prompt 与扰动构造逻辑。
- 因此要区分两件事：
  - 任务 prompt 是否能从 harness 自适应恢复。
  - 检测 prompt 是否真的对当前任务类型也成立。
- 一般来说，harness 记录型数据集的任务 prompt 可以较稳定自适应；但 `TS-Guessing / DE-COP / DCQ` 的检测 prompt 和扰动生成逻辑不保证自动跨任务通用。

## 主要文件

- [common.py](/d:/Desktop/CAT/Data%20Contamination/llm_eva/CDD-TED4LLMs/baseline_test/common.py)
  - 主入口，负责加载数据、匹配 harness task、构造 few-shot prompt、运行全部 baseline、输出汇总结果。
- `perplexity_test.py`
- `min_k_test.py`
- `reference_test.py`
- `ts_guessing_test.py`
- `decop_test.py`
- `dcq_test.py`
- `dcq_decop_test.py`

这些单方法脚本主要用于局部实验；正式运行时优先以 `common.py` 为准。

## 输入数据要求

输入数据建议至少包含以下字段中的一部分：

- `input` 或 `question`
- `target` 或 `answer`
- `is_contam`

若是多选题，推荐额外满足以下任一条件：

- 题目文本里已经包含 `A/B/C/...` 或 `(A)/(B)/(C)...` 选项
- 或单独提供 `options` / `choices`

## Prompt 相关的当前分工

### 1. 任务 prompt

任务 prompt 来自 harness task 配置，优先读取：

- `description`
- `doc_to_text`
- `doc_to_target`
- `doc_to_choice`
- `fewshot_config.samples`

这部分用于“模型本来应该怎样做题”。

如果输入数据是某个 harness 数据集的测评记录，例如：

- `bbh_date_understanding_benchmark_qwen_contam.jsonl`
- `bbh_boolean_expressions.json`
- `mmlu_pro.json`

则应优先把它视为“对应 harness 原始任务的作答记录”，并尽量通过 task 名匹配恢复原始任务 prompt，而不是重新手写题目 prompt。

### 2. 检测 prompt

检测 prompt 是为了判断模型是否见过题目，主要出现在：

- `TS-Guessing`
- `DE-COP`
- `DCQ`

这部分用于“模型是否识别出原题或原始表述”。

注意：这部分不是 harness 自动提供的内容。harness 只能帮助恢复任务 prompt，不能自动决定：

- `TS-Guessing` 该如何 mask
- `DE-COP` 该如何把原题和扰动题组成 probing quiz
- `DCQ` 该如何定义 `instance`、如何生成 perturbations、以及如何做 BDQ / BCQ

因此，`TS-Guessing / DE-COP / DCQ` 即使建立在 harness 任务渲染之上，也仍然可能需要任务级适配。

## 怎么针对数据集任务修改 Prompt 的指令

下面这些是接手时应遵循的修改指令，只写操作规则，不展开实现细节。

### 总原则

1. 先区分“任务 prompt”和“检测 prompt”，不要混在一起改。
2. 若数据来自 harness 测评记录，先假设任务 prompt 可以从 harness 自适应恢复，再检查是否匹配成功。
3. 能复用 harness 的地方，不要重写任务 prompt。
4. 只在 `TS-Guessing / DE-COP / DCQ` 明显不适配数据任务时，才改检测 prompt。
5. 改 prompt 时，优先保持“论文方法意图”不变，只做任务适配，不要顺手改方法定义。
6. 所有新 prompt 都必须支持输出稳定、可判分、可批处理，不要依赖自由解释型长回答。

### 针对任务 prompt 的修改指令

1. 先检查该数据集是否本质上是某个 harness 数据集的测评记录。
2. 若是，先检查它能否匹配到 `lm-evaluation-harness` 对应 task。
3. 若能匹配，优先使用 harness 的 `doc_to_text/doc_to_target/doc_to_choice/fewshot_config.samples`。
4. 若不能匹配，再补最小化 fallback 逻辑，不要先手写整套 prompt。
5. 若数据集是多选题，确保 `build_question_from_row` 最终输出的是“完整题干 + 完整选项”。
6. 若数据集不是多选题，先确认该 baseline 是否本来就适用于非多选题，再决定是否补任务 prompt。
7. 不要把“任务 prompt 已成功从 harness 恢复”误认为“所有污染检测 baseline 已自动适配该任务”。

### 针对 TS-Guessing 的修改指令

1. 仅在数据集天然是多选题时，优先使用“mask 一个错误选项”的版本。
2. mask 后的 prompt 必须明确要求：不要答题，只补全被遮蔽的原始文本。
3. 返回格式必须限制为“只输出缺失 span / option text”。
4. 若数据集不是多选题，不要硬套 wrong-option masking；应单独评估是否改成 token/span masking。
5. 若改成非多选版本，在文档和论文中必须标注为 adapted version，而不是 paper-faithful version。

### 针对 DE-COP 的修改指令

1. 先判断数据实例是否属于“原文片段 vs 扰动片段”的场景。
2. 若不是版权 passage 场景，不要继续使用“verbatim passage from book/paper”这类文案。
3. 对 benchmark 题目类数据，应改写成“选择与原始 dataset instance 完全一致的选项”。
4. 返回格式必须限制为单个选项字母。
5. 若数据集实例是整道题，则 DE-COP 的比较对象也应是整道题，而不是只取题干或只取答案。
6. 任何对原 prompt 的任务化改写，都应视为 adapted prompt。

### 针对 DCQ 的修改指令

1. 保留 DCQ 的核心目标：让模型识别“exact wording 的原始 instance”。
2. 先定义该数据集的 `instance` 是什么。
3. 对问答/多选题任务，`instance` 默认应设为“完整题目文本”，通常包含题干和选项。
4. 对分类/摘要/代码任务，`instance` 应按最可能泄漏 contamination 的字段组合来定义。
5. 生成扰动时，只改 wording，不改实例的任务标签、选项标签、结构边界和关键信息格式。
6. 若数据集中存在日期、公式、代码、符号逻辑表达式等高结构化内容，不要直接套一般 synonym perturbation prompt，必须单独重写扰动规则。
7. BDQ/BCQ 的问法应始终围绕“哪一个选项与原始 instance 完全一致”，不要偷偷改成“哪一个答案是正确的”。
8. 若任务类型变化较大，优先改 `instance` 定义和扰动生成逻辑，其次才改 quiz prompt 文案。

### 针对多选题数据集的统一修改指令

1. 先保证题目结构能被稳定解析出 stem 和 options。
2. 选项标签格式统一到一种规范形式后再做 TS-Guessing / DE-COP / DCQ。
3. 扰动时不要改正确答案字母本身的标签位置。
4. 如果方法检测的是“题目是否见过”，则比较对象应是题目实例，不应退化成“比较答案字母”。

### 针对非多选题数据集的统一修改指令

1. 先确认该方法是否在论文里原本支持非多选任务。
2. 若原论文不支持，不要直接宣称“自适应可用”。
3. 若必须迁移，应单独写清楚新的 prompt 假设、输出格式和判分方式。
4. 对非多选数据，优先把方法标记为 adapted，而不是 faithful。

## 使用建议

- 在新增数据集前，先人工检查：
  - harness task 是否匹配正确
  - `build_question_from_row` 输出是否符合任务原貌
  - `TS-Guessing / DE-COP / DCQ` 的 prompt 是否仍在描述正确的检测目标
- 若数据集来自 harness 测评记录，应优先验证“任务 prompt 已恢复正确”，再单独验证“检测 prompt 是否适配当前任务”。
- 若迁移到新任务后 prompt 语义已经变化，应在结果表或论文中注明：
  - `faithful` 或
  - `adapted`

## 当前已知风险

- `TS-Guessing / DE-COP / DCQ` 的检测 prompt 不保证跨任务天然通用。
- 即使 harness 已成功恢复任务 prompt，也不代表 `TS-Guessing / DE-COP / DCQ` 的 probing prompt 已自动适配该任务。
- 高结构化数据集可能导致 DCQ/DE-COP 扰动质量下降。
- 若本地代码与远端运行目录不同，修复可能没有自动同步到最终实验环境。
