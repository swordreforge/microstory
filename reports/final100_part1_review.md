# Final100 blind part 1：34篇正文复核

仅阅读本组 final100_blind_part1.jsonl 中34篇原文，并按统一正文复核标准逐篇判断。保留输入 sample_index；不依据候选标签或其他分段/模型信息。本复核为AI辅助判断，不是人工标注。

readable表示主线能否理解；language_clean要求全文无明确的非词、句法断裂、错误施事/宾语或使句意不成立的搭配；closure检查目标、相关解决行动和同目标结果；identity_consistent检查人物、关键物件及状态是否可追踪；length_ok为100–300词；combined是language_clean、closure、identity_consistent、length_ok的AND。

## 汇总

| 项目 | 通过 |
|---|---:|
| readable | 34/34 |
| language_clean | 25/34 |
| closure | 34/34 |
| identity_consistent | 32/34 |
| 100–300词 | 27/34 |
| combined | 19/34 |
| 平均词数 | 109.6 |

题材计数：ball 8篇；flower 10篇；book 6篇；birdhouse 5篇；kite 3篇；bread 2篇。题材只作描述，不参与通过条件。

边界判定：ID54、41、95等完整句重复属于冗余，不因此判语言失败；ID78的“fresh and fresh”也属重复。ID79的“Alex and Robin walked to Robin”后又写“Robin walked over to the yard”，角色的动作/位置关系不清，故language_clean和identity_consistent均不通过。ID64虽明确写面包安全到家、故closure通过，但篮子/面包的部分句子严重损坏，判language_clean与identity_consistent失败。ID29虽有多句病句，修补和放飞的主线仍能读出，readable仍通过。

## 逐篇标签

| sample_index | readable | language_clean | closure | identity_consistent | length_ok | words | combined | 主线 | 依据/边界 |
|---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|---|---|
| 90 | 是 | 是 | 是 | 是 | 是 | 116 | 是 | ball | 球滚到地上后Alex接住并再次拥有红球；重复描述不改变结果。 |
| 50 | 是 | 否 | 是 | 是 | 是 | 112 | 否 | flower | 语言失败原句：“The flower sat of dry soil was thirsty in the dirt.” 后面用杯取水，花得到足够水。 |
| 55 | 是 | 是 | 是 | 是 | 是 | 119 | 是 | flower | 小杯多次取桶水浇花，花得到足够水；重复动作仍可理解。 |
| 77 | 是 | 是 | 是 | 是 | 是 | 107 | 是 | ball | 棍子碰到球后球滚向Alex，Alex接住球；动作连续。 |
| 14 | 是 | 是 | 是 | 是 | 是 | 111 | 是 | book | 过泥路后Alex把红书交给Robin，Robin明确收到。 |
| 80 | 是 | 是 | 是 | 是 | 是 | 109 | 是 | flower | 小杯多次运水后花得到足够水；目标、行动、结果对应。 |
| 75 | 是 | 是 | 是 | 是 | 是 | 110 | 是 | birdhouse | Robin扶板，Alex用锤和钉固定，鸟屋完成并放在树上；“The nails flew into the wood”可理解为钉入木板。 |
| 54 | 是 | 是 | 是 | 是 | 是 | 120 | 是 | book | Alex两次“arrived at the door and opened it”属整句重复，不是语法错误；之后明确交书。 |
| 26 | 是 | 是 | 是 | 是 | 否 | 95 | 否 | ball | Alex用棍把球滚出并抓住；95词，主线完整但未达长度门槛。 |
| 49 | 是 | 否 | 是 | 是 | 是 | 107 | 否 | book | 语言失败原句：“They moved slowly and across the dry stones.” 交书动作和收到结果明确。 |
| 61 | 是 | 是 | 是 | 是 | 是 | 104 | 是 | flower | “The dry flower was dry in the dirt”是冗余但成立；用杯取水后花得到足够水。 |
| 79 | 是 | 否 | 是 | 否 | 是 | 128 | 否 | ball | 球已被Alex接住并找回；“Alex and Robin walked to Robin.”随后又写“Robin walked over to the yard.”，人物动作/位置关系不清，语言与身份连续性均失败。 |
| 89 | 是 | 是 | 是 | 是 | 是 | 105 | 是 | flower | 多次用杯浇花，最后花得到足够水；反复描写不影响闭环。 |
| 65 | 是 | 是 | 是 | 是 | 是 | 125 | 是 | flower | “went back to the bucket and returned to the bucket”重复；之后花得到足够水，句法可理解。 |
| 40 | 是 | 否 | 是 | 是 | 是 | 108 | 否 | birdhouse | 语言失败原句：“The board was heavy and the nails.” 其余动作仍能看出敲钉建鸟屋并放到树上。 |
| 41 | 是 | 是 | 是 | 是 | 是 | 103 | 是 | book | 两次到房子/门口是冗余，但Alex交书且Robin收到，闭环成立。 |
| 29 | 是 | 否 | 是 | 是 | 是 | 125 | 否 | kite | 语言失败原句：“Alex grabbed a large tear in the kite.” 另有“Alex smiled because the kite was perfectly.”、“The kite was a good time.”及“The kite was very in the park.”；中间写了修补并放飞。 |
| 74 | 是 | 是 | 是 | 是 | 否 | 96 | 否 | kite | 修补破洞、等胶干后放飞；96词，完整但偏短。 |
| 95 | 是 | 是 | 是 | 是 | 是 | 106 | 是 | ball | 末尾重复“Alex smiled because they had the red ball again.”是完整句重复；取球过程清楚。 |
| 64 | 是 | 否 | 是 | 否 | 是 | 112 | 否 | bread | 语言失败原句：“Alex saw the basket in the bottom of the basket.”、“Alex held the cloth and the cloth.”；目标是把面包带回家，原文明确说面包安全到家，但篮子/容器关系无法稳定追踪。 |
| 2 | 是 | 是 | 是 | 是 | 否 | 98 | 否 | book | 过泥路后Alex把书交给Robin，Robin收到；98词，主线完整但偏短。 |
| 31 | 是 | 是 | 是 | 是 | 是 | 114 | 是 | kite | 补好并放飞风筝，且写明风筝修好了；“perfectly fixed”语法成立，重复不判错。 |
| 43 | 是 | 是 | 是 | 是 | 否 | 96 | 否 | ball | 棍子把球滚出长椅下，Alex抓住红球；96词，主线完整但偏短。 |
| 53 | 是 | 是 | 是 | 是 | 否 | 92 | 否 | birdhouse | 鸟屋完成并放到树上；“gripped the board firmly against the wood”稍显生硬，但不构成明确语法或物件错误；92词。 |
| 32 | 是 | 是 | 是 | 是 | 是 | 110 | 是 | birdhouse | 重复三次“placed the small birdhouse high in the tree”，但动作和对象一致，鸟屋已建好并放树上。 |
| 23 | 是 | 否 | 是 | 是 | 否 | 97 | 否 | birdhouse | 语言失败原句：“Alex worked together for a long time.” 上文刚写Robin到场，故意图仍可推断；鸟屋完成并放置，但97词。 |
| 78 | 是 | 是 | 是 | 是 | 是 | 125 | 是 | flower | 末尾“fresh and fresh”是语法成立的重复；多次浇水后花不再干。 |
| 13 | 是 | 是 | 是 | 是 | 是 | 119 | 是 | flower | 小杯往返多次浇水，花得到足够水；重复提到取水仍可理解。 |
| 34 | 是 | 是 | 是 | 是 | 是 | 110 | 是 | book | Alex过泥路、到门口敲木门，Robin开门并收到书；“tapped on the wood”语境可理解。 |
| 87 | 是 | 是 | 是 | 是 | 是 | 116 | 是 | flower | 小杯多次运水后花得到足够水，花显得新鲜；闭环明确。 |
| 25 | 是 | 是 | 是 | 是 | 否 | 96 | 否 | ball | 棍子把球滚出长椅下，Alex接住；97词，主线完整但偏短。 |
| 94 | 是 | 是 | 是 | 是 | 是 | 113 | 是 | ball | 棍子把球推出并由Alex抓回；Robin走到草地不构成角色或物件矛盾。 |
| 48 | 是 | 否 | 是 | 是 | 是 | 121 | 否 | flower | 语言失败原句：“Alex took many small cup.” 其余浇水过程和花得到足够水的结果清楚。 |
| 97 | 是 | 否 | 是 | 是 | 是 | 103 | 否 | bread | 语言失败原句：“Alex grabbed a basket and then at the bakery.” 后续找到布、包面包并把面包带到家，结果明确。 |

不通过项：language_clean 为sample_index 50、49、79、29、64、40、23、48、97；identity_consistent 为64、79；100词以下为26、74、43、53、23、25、2。ID79在语言与身份两项都失败；ID64语言与身份失败，但目标和送达结果仍可追踪。

以上比例只描述这34篇盲评样本，不代表总体生成质量。
