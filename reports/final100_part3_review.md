# Final100 盲审第三组正文复核

本报告只依据 `final100_blind_part3.jsonl` 内的 33 篇正文逐篇完整复核，保留输入的 `sample_index`。按 [`20k_review_criteria.md`](20k_review_criteria.md) 标注。这是 AI 辅助复核，不是人工标注；没有读取其他分段或模型元数据，也没有调用 Gemma、修正文或推断缺失事件。

| 指标 | 通过数 |
|---|---:|
| readable | 32 / 33 |
| language_clean | 29 / 33 |
| closure | 32 / 33 |
| identity_consistent | 30 / 33 |
| length_ok（100–300 词） | 28 / 33 |
| combined（clean、closure、identity、length 全通过） | 24 / 33 |

失败样本均以输入中的 `sample_index` 标识：

- **86** 是唯一 `readable`、`closure` 未通过的样本，且 `language_clean`、`identity_consistent` 也未通过。原文从 “a small hole in the bottom of the basket” 转到 “The paper was a large tear both in the basket”，再到 “Alex saw the warm and found a clean patch” 和 “Alex ran the door and held the bread into the blue sky.”，结尾才出现 “The kite flew again.”。主体对象和目标无法稳定追踪，不能从文本确定一个连续的故事目标、解决行动和结果。
- **46** 语言不干净： “The kite was bright and a large tear in the paper.” 缺少连接后半句的谓语。开头篮子破洞与后文风筝修理不相关，但没有声称它们是同一物件；因此只记为无关开场，不另判 identity 失败。后面仍明确修补风筝并让它飞起，closure 通过。
- **11** 语言和 identity 未通过。结尾 “The flower got enough water and poured Alex.” 把花写成向Alex倾倒；此前主线是Alex给花浇水。前文已明写花得到足够水，因此 closure 仍通过。
- **37** 语言和 identity 未通过。 “Alex had a good and held the bread inside the blue sky.” 是断裂句，也把前文在篮子里的面包改写成在蓝天里；最后 “the bread reached home safely” 明确写出送回家结果，所以 closure 通过。

低于 100 词的 5 篇为 **70（92）、96（96）、3（84）、39（98）、27（97）**；其余处于 100–120 词。它们的目标和结果能分别追踪的仍通过 closure，但不满足综合条件中的长度要求。

完整句或动作重复不自动视为语言失败。例如 sample_index 76 重复 “Alex had the red ball again”，69 和 93 重复跨石步行或交书，均保留其余通过标签。`house of Robin`（sample_index 63）虽生硬但可理解，singular they 保持接受；边界处按明确错误和对象可追踪性判断。

逐篇标签及每项失败的同篇原文依据见 [`final100_part3_labels.jsonl`](final100_part3_labels.jsonl)。
