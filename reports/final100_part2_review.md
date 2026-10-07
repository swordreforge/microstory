# Final100 blind part 2：逐篇正文复核

本报告只依据 `final100_blind_part2.jsonl` 中的 sample_index 与正文，未读取其他分段、模型元数据或旧结果。按统一标准逐篇标记 readable、language_clean、closure、identity_consistent、length_ok，并以四项 AND 计算 combined。词数由正文空格分词计算。primary_plot 仅为题材诊断，不作为质量门槛。这是 AI 辅助复核，不是人工标注或客观成功率。逐篇机器可读标签见 [`final100_part2_labels.jsonl`](final100_part2_labels.jsonl)。

| 指标 | 通过 / 33 |
|---|---:|
| readable（主线可辨认，允许少量局部错误） | 32 |
| language_clean（全文无明确语言错误） | 26 |
| closure（目标—相关行动—同目标结果） | 32 |
| identity_consistent（人物/施事/关键物件状态稳定） | 29 |
| length_ok（100–300词） | 27 |
| combined（语言、闭合、身份、长度同时通过） | 22 |

| sample_index | primary_plot | 词数 | readable | language_clean | closure | identity_consistent | length_ok | combined | 原文证据/边界 |
|---:|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 68 | book_return | 106 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 88 | ball_retrieval | 104 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 58 | book_return | 102 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 66 | birdhouse | 110 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | 前文写“They needed to make a hammer”，之后却直接“Alex took a hammer”；关键工具从待制作变成现成物，未交代状态变化。 尽管 hammer 的计划/来源有跳变，建成鸟屋并放置的主线仍能辨认。 |
| 6 | book_return | 116 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 81 | flower_watering | 119 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | “carried many cups of water”按多次 cupfuls 理解，不视为突然出现多个新杯子。 |
| 0 | ball_retrieval | 114 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | “Alex tried to help out from the bench”表达别扭，但可理解为在长凳处帮忙；未据此单独判错。 |
| 17 | kite_repair | 114 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | “The kite was sad in the paper”带有拟人且介词用法生硬，后续修补与放飞对象仍可追踪。 |
| 60 | mixed_unclear | 91 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | “The hole was warm and a large tear”; “Alex wanted to fix the bread on the bakery.”把篮子破洞和面包目标混在一起，关键对象/动作关系无法成立。 开头目标写作“fix the bread on the bakery”，正文却转为包面包并带回家；不能将两种不同目标补成同一任务。 根据正文空格分词计数为 91，低于100词。 |
| 10 | flower_watering | 109 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | “Finally, the water was flat.”把水描述为 flat，不能表达本段浇水结果。 |
| 44 | birdhouse | 93 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 根据正文空格分词计数为 93，低于100词。 |
| 16 | birdhouse | 100 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 12 | flower_watering | 117 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 18 | book_return | 119 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Robin 说“I brought help!”略突兀但语法成立；Robin 随后明确收到书。 |
| 42 | birdhouse | 95 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 根据正文空格分词计数为 95，低于100词。 “The nails arrived at the new home”把钉子拟作回家，结合建鸟屋可理解为钉子进入木结构。 |
| 67 | birdhouse | 84 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | “The birdhouse was sturdy and safely.”sturdy 与 safely 词性不同，不能这样并列。 根据正文空格分词计数为 84，低于100词。 |
| 47 | flower_watering | 108 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 98 | flower_watering | 98 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 根据正文空格分词计数为 98，低于100词。 |
| 83 | book_return | 108 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 22 | book_return | 109 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | “Robin smiled and said, “I brought the red book,” Alex said.”同一句把台词同时归给 Robin 与 Alex。 交书人明确是 Alex，但同句又将“I brought the red book”归给 Robin，随后再写 Alex said，人物发言/交接角色不一致。 |
| 8 | flower_watering | 112 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | “Alex carefully poured the water until the flower book was empty.”把 flower 与 book 混作容器对象，句意不成立。 已使用的关键容器是 cup，结尾却称“flower book was empty”，关键物件身份发生混淆。 |
| 20 | flower_watering | 106 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 73 | flower_watering | 121 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | “I brought many water.”many 与不可数名词 water 不匹配，且后续说话者/句子结构混乱。 “I brought many water”没有明确说话者，随后 Robin/Alex 的发言归属也混乱，人物发言身份无法稳定追踪。 |
| 38 | book_return | 110 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 33 | flower_watering | 117 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 85 | flower_watering | 112 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 71 | book_return | 113 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 21 | ball_retrieval | 105 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 4 | birdhouse | 95 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | “They worked together bucket.”是句法残片。 根据正文空格分词计数为 95，低于100词。 |
| 62 | kite_repair | 102 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | “waited for the glue to get sturdy”不自然，但同篇还明确写了等待胶水变干；不单因风格问题判错。 |
| 9 | book_return | 113 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 45 | flower_watering | 123 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |
| 1 | book_return | 119 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 主线目标、相关行动和结果可追踪；无明确语言或身份错误。 |

## 汇总与边界

本组共 33 篇。readable 32/33、language_clean 26/33、closure 32/33、identity_consistent 29/33、length_ok 27/33；严格 combined 22/33，统计由本组逐篇标签计算。sample 60 的开头目标是“fix the bread on the bakery”，正文却包好面包并送回家，不能把它补写成明确的“把面包带回家”。其余 32 篇主线可辨认。

closure 未通过 sample 60。language_clean 未通过 sample 4、8、10、22、60、67、73；identity_consistent 未通过 sample 8、22、66、73。词数不足100词的 sample_index 为 4（95）、42（95）、44（93）、60（91）、67（84）、98（98）。

边界记录：sample 17 的 “The kite was sad in the paper” 是拟人且表达生硬，但后续 kite 与 paper patch 可追踪；sample 62 的 “glue to get sturdy”不自然但同篇又写等待干燥，未单独判错；sample 81 的 “many cups of water”按多次 cupfuls 理解；sample 42 “nails arrived at the new home”按儿童故事拟人处理。singular they、重复句、book received 明确交接均未自动扣分。sample 66 的 hammer 状态跳变、sample 22 等由错误的发言/行动者归属判身份失败。以上为盲评正文的 AI 辅助判断。
