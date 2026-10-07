# 约20k共享循环故事模型的持续研究

用户最新目标：最终候选保持约20k独立参数、looped、自回归从BOS生成100–300词完整故事，重点语言、连续性和可读性，支持HFcustom code加载。允许数据/词表/架构/目标/优化器创新；每次观察实际正文再调整，直到多个方向与种子复验显示收益低。不能把测试通过或EOS当完整性。历史较大模型只作记录/可选蒸馏teacher，不参与20k最终排名；最终只保留最佳权重，配置/日志/所有输出保留。

先检索2025–2026 primary来源，三个子代理分别架构/数据词表/目标优化器。R18 legacy接近预算基线V448H32I64=24769，R4/value.5，fresh3257/B32/Muon.01/16k/seed42，CE1.116417，T.8 40样本readability.025/4，严重乱码/混plot，尚无可用能力证据。

R19 nativeV448H24I96=20113与factorV1024H32I64r8=20185建立20k分配基线。两者单physicalR4/4heads/value.5/Muon.01/B32/16k/seed42，新3257train与200val/200test相同文本，词表仅train学习。最终CE/BPB+同条件T.8正文，后续T.5解码单独记录。词表/hidden/rank配方组合改变，不声称单变量。

## R19 实际结果与下一步依据

两个候选已完成训练和两档各40篇采样，均只从BOS开始。基础训练使用3,257篇、200篇验证、200篇测试，故事曝光510,908次。BPB在同一200篇验证正文上计算，人物槽换成固定姓名后的UTF-8正文142,503字节；EOS NLL计入，BOS不预测。不同词表的CE不能直接排名。

| 候选 | 独立参数 | 最终Trainer验证CE | token加权验证BPB | T=.8 可读性AI均分/4 | T=.5 可读性AI均分/4 | T=.5 重复4-gram比例 |
|---|---:|---:|---:|---:|---:|---:|
| native V448 H24 I96 | 20,113 | 1.239687 | .745575 | 0.000（39有效评分） | .200（40有效评分） | .048666 |
| factor V1024 H32 I64 r8 | 20,185 | 1.875712 | .684388 | .076923（39有效评分） | .500（38有效评分） | .038787 |

两候选的T=.8/.5各40篇都达到100–300词且正常EOS；四项评分均>=3的诊断计数均0。根代理抽读0、1、2、3、10、20、30等原始样本，仍有非词、语法片段和跨情节物件/动作混合。降低温度略改善AI可读性分数，却提高重复率，不能据此称模型可用。逐篇严格AI审阅另由training_eval完成。这些评分和代理判断均不是人类标注或客观成功率。

24,769参数历史近预算候选在T=.5也没有四项均>=3样本（0/40），可读性.230769/4（39有效评分）、重复4-gram.040350。这只补充解码诊断，不参与最新约20k排名。

### R20 学习率对照（已完成）

固定R19 native的20,113参数、V448/full256bytealphabet、H24/I96、physical1/R4/4heads、value residual .5、3,257篇训练、seed/data_seed42、B32、16,000更新、cosine/5%warmup、每2,000步验证与保存。只将Muon LR .01改为.003或.03；其余设置不变。依据是R19训练与验证仍下降，且训练/验证差距小；先判断优化设置能改善多少语言，再将词表/数据变化另作实验。

两组现已完成训练以及T=.8/.5各40篇采样。LR=.003最终CE1.286757、同200篇验证BPB.772901；LR=.03最终CE1.212831、BPB.729144。原LR=.01的CE1.239687/BPB.745575位于两者之间。较大学习率改善了似然，但T=.8 AI可读性仅0/.025（各40有效评分），T=.5仅.025/.125；两温度的四项均>=3诊断计数均0/40，未出现可交付质量证据。T=.5重复4-gram分别.054802/.063893，不能用更低温度掩盖语言问题。逐篇AI审阅单独记录。这里尚未做训练种子复验，不能把单种子小幅似然差异当稳定质量收益。

### R21 观察字母表与预算分配（已完成）

两候选使用相同3,257篇训练、200/200 heldout与LR=.01/B32/16k/seed42/R4/value.5；仅TRAIN学习observed alphabet，使用明确UNK且V内计数。observed V448 H24 I96=20,113与R19 native架构相同，主要检验词表构造；observed V320 H32 I56=19,905额外把预算转给hidden width。V448训练token减少23.3%、V320减少10.7%，两者heldout400篇全部exact roundtrip且UNK0。相同训练故事曝光510,908次，但token曝光/序列长度及按长度组batch的具体次序会随tokenizer改变；不称固定FLOPs对照。对应[CPU tokenizer探针](observed_alphabet_probe.md)。

最终V448的CE1.341930、BPB.620988；V320的CE1.166238、BPB.627235。在同一142,503字节的验证正文上，V448相较旧full-alphabet V448的BPB.745575明显下降。T=.5严格逐篇AI审阅最终A/B/C分别为1/37/2、0/26/14，两组全文语言合格数均0。V448 sample12的“取书—踩石过泥—到朋友家—交还书”主线可读，但包含`wanted to resents`等局部错误；sample34缺少原文可支持的回家行动，已从A改为B，不能替模型补写事件。T=.5重复4-gram比例分别.077970/.046518。尚无训练种子复验，不能称稳定能力。

### R22 循环深度向量与MQA预算分配（训练与两温度采样已完成）

都使用observed V320、相同3,257篇TRAIN和200/200 heldout、physical1/R4、value residual .5、Muon LR=.01/B32/16k、seed/data_seed42。loopindex保留H32/I56/KV4，仅增加零初始化的[R4,H32]深度向量，参数20,033；mqa_ffn禁用深度向量，将KV4改为KV1、I56改为72，参数仍19,905。这是两种独立假设，不将MQA/FFN组合解释为单变量。

loopindex的CE1.156140/BPB.621880；mqa_ffn的CE1.180160/BPB.634747。主对照R21 V320 BPB.627235。loopindex在T=.5的38份有效Gemma评分中可读性.657895/4、重复4-gram.038275，39/40在长度范围，1份耗尽2047新token而未EOS；mqa_ffn的37份有效评分可读性.594595/4、重复.053131，40/40长度及EOS正常。两组四项均>=3的诊断计数均0。[实际80篇逐篇AI审阅](r22_20k_prose_review.md)已完成：loopindex A/B/C=0/31/9，mqa_ffn=0/33/7，两组全文语言合格数均0/40。loopindex截断样本12为933词。两个候选均没有提供主要故事可读且闭合的质量收益，不能仅据CE或终止率保留新架构。

### R23 输入再注入与物理层数（已完成）

第一候选为R22 MQA的19,905参数、V320/H32/I72/physical1/R4/KV1，只增加无参数原输入混合，loop2/3/4的系数为.25/.166667/.125。输入不额外归一化且保留梯度，不叠加循环向量。第二候选V320/H24/I52/physical2/R2/KV4为20,017参数，总展开深度4；这是宽度、FFN、物理层多样性的配方比较。共同训练设置不变，CPU完整/增量cache输出最大差均8.94e-8，参数计数实测。HF重载与梯度检查见[input reinjection记录](input_embedding_injection_check.md)。[LOOM](https://arxiv.org/abs/2610.01153)只提供机制动机，100M–1.7B MoE完整配方未移植，不声称复现论文或20k收益。

### 300表面词形语料：先观察失败再调整

300项精选词形只从训练子集学习，覆盖现TRAIN普通词token81.19%，不是频次top300的89.01%口径。初次prompt-only 192请求，117篇格式接受、75篇拒绝、0请求错误；188/192总稿有OOV，117格式接受中116有OOV，只有1篇真正零OOV。不能把格式接受当封闭词表合格，也没有将该pilot加入正式训练。

查[vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)与[Gemma4 reasoning/regex支持](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)后，新增教师可选`--constrain-word-list`。regex只约束整词与人物槽/标点边界，不指定句子或情节；默认请求不加约束，学生训练/自由生成无约束或外部文本修复。最小六情节协议探针均正常finish、OOV0且fullmatch；其中4篇120–160格式接受，2篇过短（97/116词）。实际API prompt_tokens1216–1230，completion含16reasoning。原名单缺happy/sky等词，出现“felt very big and bright”“felt very much much”“blue sun”等失真；因此先保留v1原始名单和输出，再另建train-frequency+必要情节词的v2，进行同名单/同seed的prompt与regex配对数据试验，尚不扩大生成规模。

### 当前策略优化判断

Gemma反馈属于RLAIF；没有人类偏好标签，未完成RLHF。R19的80篇T=.5的逐篇AI审阅为native A/B/C=0/15/25、factor=0/22/18，两组无明显语言错误的整篇数均0/40。基础英语尚未形成时，直接做GRPO可能大量组优势为0，也可能优化Gemma假阳性。先做SFT/词表/数据；若出现可读候选，再用64rollouts诊断奖励差异，old rollout policy的ratio与fixedSFT reference KL必须分开。研究依据与限制见[20k训练方法研究](20k_training_research.md)。

### 并行的数据/架构准备（尚未正式训练）

训练前语料池3,657篇有1,871种小写普通词形，top300覆盖89.0%；正式词表必须仅从3,257篇训练子集学习。V448池统计中201个基础区间ID未见于正文，而高于基础区间的192个ID中177个已使用，所以不能说大多数learned merges浪费。先测试train-observed byte alphabet+显式UNK的V320 BPE，报告heldout未知字符与roundtrip；并用Gemma小批试生成仅约300–350种表面词形的同长度/同六情节故事，统计OOV与实际语言质量。两阶段SuperBPE先后子词/跨词合并的CPU探针后置，没有质量结论。

前沿依据包括[Looped GPT-BERT, 2026-09](https://arxiv.org/abs/2609.09691)、[DeepLoop, 2026-07](https://arxiv.org/abs/2607.13491)、[TokSuite, ICML2026](https://proceedings.mlr.press/v306/altintas26a.html)、[SuperBPE, COLM2025](https://arxiv.org/abs/2503.13423)。这些研究没有20k BOS完整故事的成功证据。Looped GPT-BERT论文的最终模型是12.18M/physical4/R12；[quesswho公开looped实现](https://github.com/quesswho/looped-transformer)为另一个约50M/physical8/R2研究实现，不能当作该论文作者代码或混合规模。

### V2/V3教师词表反馈与整词分词准备

V2为300项TRAIN已出现词形，覆盖87.63%的训练普通word tokens。配对96请求prompt/regex分别52/50篇120–160格式接受，但prompt只有4篇接受稿零OOV；regex全部零OOV。V2缺stay/feel等基式，产生`to stayed`、`did not trips`、`They were very carefully`，零越界仍不等于语言合格。100–119词且只因120门槛被拒的33篇regex输出可额外检查，原始接受状态保持不变。

V3保留V2并加入50个TRAIN实际词形，共350项，覆盖88.51%，尚未添加TRAIN未出现的does等形式。同seed50000000/sixplots各16、官方T1/top_p.95/top_k64/思考16，regex96请求已完成：65篇120–160格式接受、28篇过短、3篇耗尽输出且超长、0API错误。全部接受稿OOV0；两篇截断稿末尾有未完成词形，raw总OOV3。教师prompt实测1319–1330tokens，故本次max_tokens640，V2为768；这是必要上下文调整，不声称完全相同请求预算。所有raw和正文保留，尚未进入正式TRAIN，逐篇语言质量检查进行中。

整词WordLevel探针使用官方Tokenizers/ByteLevel并保留原始大小写、标点和人物原子槽；实际tokenizer词表并不等于350普通词形。若heldout含UNK，预测的是替换后的文本，无法用该似然与原始正文比较。evaluation/likelihood.py现报告unknown_token_count，并对有UNK的输入将BPB置为null。一次WordLevel18/1/1隔离探针的验证稿有15UNK，因此不输出原始正文BPB；训练18篇UNK0仍可正常计算。这是指标适用范围修正，不是质量改进。

前沿检索还查阅[Learning syntax without semantics, ICML2026](https://proceedings.mlr.press/v306/winston26a.html)：在保留句法、抑制语义的改词训练上观察到5M与30M模型的样本/参数效率差异；完整故事需要物件和行动语义，不能直接照搬语义置换。[Attractor Models, 2026-05](https://arxiv.org/abs/2605.12466)采用backbone预测后固定点精炼与隐式梯度，其语言实验140M起、推理27M，不是20k证据。[Latent Recurrent Transformer, 2026-09 revision](https://arxiv.org/abs/2605.26797)将上一token隐状态注入下一token并用交错并行训练，实验1.3B/2.1B；本项目未实现跨token递归。这些方向先记录适用条件，当前仍优先解决已观察到的坏词形和不可读语料。

### V4小试与受控语料第一批

V4实际359项，去掉普通裸词s并加step/steps/longer/any/another/we/bird/birds/sips/cups；全部新增来自TRAIN。旧人物所有格提取多算321个s，修正后普通token409,257/types1842，V4覆盖88.70%，V1–V3历史文件保持不变，见[vocab v4](vocab_v4.md)。新seed51000000、word-range100–140/aim120、max640、官方采样/思考16的96请求：62格式接受、34仅过短、0API错误/OOV/重复；62篇全部逐篇读，57语言可接受、62主线闭合，坏5篇均有完整原文依据。不同seed和词表/长度目标，不作单变量因果解释。

V3教师质量rubric probe保留83/91，其中一些高分漏掉明确搭配错误；原AI标签也有两处误引，已从raw核实并追加更正。Gemma自动筛选未作为单一正式TRAIN门槛。新stage1按V4词表生成516请求=sixplots各86，seed52000000/concurrency24，word-range100–160/aim130，max640、思考16、官方T1/.95/k64不变；依据是aim120时34/96过短。所有正文/拒稿保存，子代理分批阅读全文并记录language_ok/closure_ok及同ID原句，不修改原文。计划仅用两项合格稿与pilot57合格稿精确去重，建立新的TRAIN/40val/40test；不混旧corpus。仍未进行新语料正式训练。

WordLevel已修复实际发现的训练姓名所有格边界：ASCII [PERSON_NAME_1]'s在trainer被拆成 ]'与s，而运行时留下's，因此原73篇TRAIN有4UNK。最小预分词边界修正后73/73准确往返、TRAIN UNK0，实际V339；heldout仍有真实未见表面词/标点UNK，不掩盖未知词且BPB为null。BPE路径未变。

### R23结果及审阅尺度核对

输入再注入19,905参数的CE1.167892/BPB.627896，双物理层20,017参数的CE1.154770/BPB.620505；都未改善全文语言合格数（T=.5各0/40）。原始逐篇AI审阅A/B/C分别13/13/14和20/17/3，但A允许局部错误的尺度可能与此前审阅不同，已安排R21V448/R22MQA/R23physical2各前10篇混排核对；不把13/20与此前0/1直接作收益排名。两组T=.5有效Gemma评分38/37，可读性.6842/.7838，四项>=3均0，重复4-gram.056368/.038566。

受控V4最后选出57+351=408篇语言及闭合双合格原文，合并精确重复0；新whole-story split seed42：TRAIN328/val40/test40。observed BPE V320 +BOS/EOS后token66322/8237/8129，WordLevel实际V378为46902/5788/5718，约减少29%的序列token。WordLevel上限768只是trainer允许容量，未使用空词项补齐。实际UNK/往返和参数计数正在进行必要输入检查，随后启动BPE H32/I72/MQA=19,905与WordLevel H32/I53/MQA=19,937的两候选。共同physical1/R4/value.5/MuonLR.01/B32/16k/seed42，不把FFN/词表预算配方组合声称单变量tokenizer实验，学生自由BOS生成无regex限制。

### R24 新受控数据结果与R25早期学习率扫描

R21/R22/R23前10篇混排统一口径复核已完成，30篇中18篇旧标签改变；统一口径A/B/C分别7/2/1、4/5/1、5/5/0，但30篇全文语言干净均0。旧全量A0/1与R23A13/20不得直接比较，也不能将30篇比例外推到全量。见[rating calibration](r21_r23_rating_calibration.md)。

V4新数据实际检查：BPE320全部408篇零UNK/exact；WordLevel实际378，TRAIN328全部零UNK/exact，val3UNK/37exact、test2UNK/38exact，均为训练未见表面词。两模型实际19905/19937参数，[输入复核](controlled_v4_prepared_check.md)。新40篇val原始marker正文29093bytes，固定姓名渲染正文24293bytes；不能混用字节表示。

R24两候选16k已完成，但最早保存step2000为最佳，之后验证CE逐次上升：BPE1.092→1.407、WordLevel1.491→2.664。导出best2000模型，最终重新验证CE1.092828/1.490396；CPU token加权BPE BPB.531618，WordLevel heldout3UNK故原文BPB null。和旧3257/200数据指标不能直接比较。T=.8 Gemma可读性.3333/1.359（各39有效），四项>=3为0/5，长度合格35/37；T=.5为1.0513/1.6053（39/38有效），四项>=3为5/6，长度37/36；全部正常EOS。这些是模型诊断不是人类成功率，实际逐篇审阅进行中。根代理抽读正文发现整词消除频繁非词，但病句、对象错配与重复仍明显。

R25在WordLevel19937/相同328/40/40上扫描MuonLR.003/.01/.03：同4k总步数/每250步验证与保存、B32/5%warmup/cosine/seed42/data_seed42；架构H32/I53/physical1/R4/MQA/value.5不变。依据是R24过拟合与早期best检查点过稀，假设更早checkpoint改善泛化。三组内部仅LR变，与R24同时改变schedule长度和验证频率，不声称跨轮单变量；launch .003/.01先双并行，.03接续。

### R26循环训练目标与计算量（已启动）

沿用R25 .003WordLevel19937/V378/H32/I53/MQA/physical1，两个独立对照：aux01为R4、训练最终CE+.1×早期三循环同next-token CE均值；loop8为R8、aux0。相同LR.003/B32/4k/eval250/5%warmup/seed42/冻结328/40/40，不增加权重。aux实现已验证训练公式和eval不加aux；R8实际count19937/fullseq-vs-cache最大差9.69e-8。新研究见[algorithm report](20k_next_algorithm_research.md)。Apple LatentLookahead需要latent上下文位置与双向组mask，不移植半套为loopCE。

### R27自主类型token成对对照（已启动）

依据是实际R24/R25正文将bread/kite/book等物件混接，[故事结构研究](20k_story_type_research.md)强调一个类型token不是完整计划。实验prepare_story_type.py从teacher原稿全文精确join类型，不修改408篇正文/split。两臂均从旧WordLevelV378添加6TYPE特殊token到V384，旧ID不变；H32/I51/MQA/physical1/R4/value.5实测19937，LR.003/B32/4k/eval250/5%warmup/seed42。plain训练BOS->body，type训练BOS->TYPE->body，都从BOS-only自由生成。三split正文tokens去前缀后与原token编码完全一致；每篇仅多1原子TYPE。baseline也有相同6unused词表行和相同FFN，所以内部保持参数/词表匹配。

renderer只隐藏首个注册TYPE，误生成在中途的TYPE保留并评为正文错误；raw_text保存原前缀，评估另记story_type_prefix；无姓名的类型样本也保留原文。实际构造样例确认中途TYPE保留、姓名照常Alex、无姓名前缀raw可见。不使用外部类型提示/固定句子/语法修补；decoder标准HF接口不改。含TYPE的验证CE含新增类型预测目标，不能直接据此与plain正文CE排名；本轮以原始正文语法/闭合/一致性和类型对应为主要证据。

### R25正文结果、解码与R28

R25 .003/.01 T=.5各40篇完整审阅：[prose report](r25_20k_prose_review.md)，language_clean5/2、closure34/30、identity37/39、长度38/35、combined5/2；根代理另全文确认所有7个联合正例。LR.03的40篇待同口径追加。新温度.3同权重40篇根代理读完，明确联合正例3/40、词数32/40、题材28/40还书且0花、重复.0560高于.5的.0345；不采用为默认，见[解码报告](r25_20k_decode_t03_review.md)。

R27两臂T=.5各40篇：[完整正文审阅](r27_20k_prose_review.md)，均language_clean0/combined0，plain22/type28在词数范围；TYPE40/40有效却仍出现prefix与正文类别冲突、跨情节混写。此单seed实验不支持保留TYPE改动。

R28已完成训练：GELU真two-matrixV378/H32/I80/MQA/R4/value.5=19969，MuonLR.003；另SwiGLU原19937/I53，仅优化器改AdamWLR.003。固定数据328/40/40/B32/4k/eval250/5%warmup/seed42。最佳实际保存step1000/1500（最后4k eval是恢复best后的再验证，不是best保存位置）；恢复CE1.344770/1.239208。只据CE观察AdamW值得读正文，T=.5无judge生成40各已完成，完整审阅进行中。兼容实现见[ungated FFN](ungated_ffn_check.md)。

Stage2扩充生成实际1032requests/734accepted/293reject/5HTTPtimeout，所有接受稿0OOV且无精确重复；1027raw，拒因282短、11截断且超长，5无响应不重试。多组Gemma评估并发造成126等待/9running、teacher部分超时，已经停止叠加高并发judge并改后续先保存学生正文；目前队列0。734格式接受稿正在全文审阅，尚不作为734合格TRAIN宣称。冻结WordLevelV378在这些稿上690零UNK/44篇51UNK，都是新表面词项而非语言失败；下一数据规模纯对照可先在语言闭合合格稿中选零UNK原文追加TRAIN，保留原40/40 heldout与原词表。


### R26–R30正文复核、数据扩容与重新扫描

R26早期循环CE辅助与R8在固定328TRAIN上的T=.5严格综合为0/40、1/40；R27成对TYPE实验均0/40。R28此前宽口径可读正例含明确错句，已在逐篇表增加严格language_clean并修正：GELU与AdamW均严格clean/combined0/40，不能用旧1/3可读数作为严格质量。AdamW词表实际V378，见各轮正文报告。

Stage2 734格式接受稿全文AI审阅：688语言合格、734闭合；两项合格中39篇包含既有WordLevel未见token。保留649零UNK新稿，追加原328TRAIN成为977篇，旧V378 tokenizer、40validation/40test冻结，TRAIN共139,483tokens含BOS/EOS。没有把零UNK当语言合格，所有734稿、293格式拒稿与5超时记录均保留。见controlled_v4_stage2_review.md和controlled_v4_scale_counts.json。

R29为纯数据规模对照：19937/V378/H32/I53/MQA/N1/R4/value.5/LR.003/B32/4k/eval250/seed42，分别Muon、AdamW，977/40/40。恢复最佳CE .968890/.980557，实际最佳保存4000/3750步。严格全文AI审阅40篇各：Muon clean24/combined21、AdamW clean23/combined14，词数合格34/27，EOS都40，重复4gram均值.02445/.02813。Muon ID16原审阅漏掉“grabbed a large tear”，根代理阅读全文后剔除，报告和总数已修正；根代理全文核对全部21个Muon综合正例（包括后续统一尺度接受的ID9）。相较原328TRAIN的R25 Muon5/40，这是一项明显但单seed、固定40生成样本的改善，非人类正确率。训练集仍限六种情节，不能据此声称开放域故事能力。

R30是两种预算配方比较，固定977/40/40、Muon.003/B32/4k/eval250/seed42/N1/R4/value.5：MHA V378/H24/I116/q4/KV4=19873，factor V378/H32/I101/r16/q4/KV1=19947。恢复CE .958476/1.016873，最佳实际4000/2750。同V378正文预测目标可比较CE，但不能据CE替代整篇质量；T=.5原文各40正在严格复核。前者把预算转给FFN和value宽度，后者扩FFN但限制输出rank16，因此不能解释为一个单变量结论。

R31已启动Muon学习率重新扫描.001/.01，和R29 .003只改LR，其余19937/V378/H32/I53/MQA/N1/R4/value.5/977TRAIN/4k/B32/eval250/cosine5%warmup/seed42不变。Stage3再1032教师请求，seed54000000/concurrency12/官方T1/top_p.95/top_k64/thinking16/max640/100–160词，复用V4wordlist与六plots，结果尚未用于训练。

前沿n-gram记忆研究见20k_ngram_memory_research.md：Engram、Lngram v2为大规模证据，61×67双哈希微型探测的二元组类型碰撞59.44%、位置53.05%。1536参数表+投影需将I53降I37才能保持19937，当前不直接替换基线；未实现或训练此模块。训练种子复验、独立解码样本与最终权重整理仍未完成，尚未满足收益低停止条件。


### R30–R34结果与Stage3实际数据

后续统一尺度更正后，R30 MHA/factor的严格综合为17/40、15/40；R31 Muon .001/.01为4/40、18/40，均未超过R29 .003的21/40。R32继续训练8k与双物理层4k分别为24/40、25/40，恢复验证CE .957136/.941625，两个配方都仍有明确局部语法及对象状态错误。

R33弱循环辅助CE .05、MQA两查询头、GQA两KV头分别为16/40、22/40、23/40；验证CE .984311/.934513/.950216。降低CE没有自动变为较高的正文严格合格数。R34把977篇按六情节各210记录重复采样，得到1260训练记录、仍为977唯一篇；严格语言25/40、闭合38/40、一致性38/40、长度37/40、综合23/40。该单种子结果不支持采用题材均衡重复采样。

long8k的HF原生min_new_tokens=152生成对照使长度从30/40提高到40/40，但语言从31/40下降到16/40、综合从24/40下降到16/40，重复完整句与4-gram同时增加。保留测试记录，不采用强制长度下限。

Stage3实际1032请求、768格式接受、264拒绝、0API错误。三代理分段阅读全文：语言693/768、闭合768/768、身份765/768，三项同时690/768；其中36篇含冻结WordLevel未见token。最后新增654篇，原977篇全部保留，TRAIN成为1631篇、233756tokens，V378与40/40 heldout固定。新批还显式筛身份一致性，因此不声称只改变数据大小。详见controlled_v4_stage3_scale2_prep.md。

R35已启动在1631TRAIN上比较三个配方：native19937/H32I53/N1R4/q4KV1；physical2 19969/H24I54/N2R2/q4KV1；mqa2 19969/H32I48/N1R4/q2KV1。全部Muon .003、B32、4k、eval250、seed42、cosine5%warmup、value residual .5、无aux，逐篇40样本T=.5。训练种子复验、独立解码样本和最终权重整理尚未完成。


### R35正文结果与R36–R37针对性调整

R35 native/physical2/MQA2的恢复验证CE分别.907899/.891582/.874070；严格综合22/24/25，语言27/29/33，闭合40/40/38，一致性37/40/37，长度33/35/33。相较977篇对应配方，新增654篇的预测收益明显，正文联合计数收益较小，不能按CE最低直接选择。

R36继续在MQA2上仅改LR为.002/.005，或保持.003改R8。前两者CE .888970/.862046、严格综合26/27、语言30/31、闭合39/40、一致性38/38、长度33/36；改善仅1–2篇，尚不构成稳定提升。R8恢复CE .904637，正文复核待完成。三者均19969、1631TRAIN、B32、4k、eval250、seed42。

按正确显示姓名口径，1631篇TRAIN均值112.8词、中位112；≥120/130/140为363/60/0。此前直接对PERSON_NAME marker计词的135.0均值已作废。来源应包括regex_pilot及stage1/2/3，完整exactbody join无unknown。R37只让现有363篇≥120词稿各重复一次，1631unique/1994records、平均115.0词；原MQA2/.003/R4/19969/4k/B32/seed42不变，既有tokenizer/heldout固定，学生不设置EOS下限。结果待完整正文复核。准备脚本experiments/prepare_length_weighted.py，真实统计controlled_v4_length_weighted_counts.json。


R36 R8全文结果已完成：语言30/40、闭合39/40、一致性39/40、长度30/40、综合22/40，不保留。R37作者最终标签为语言33/40、闭合39/40、一致性34/40、长度33/40、综合27/40；先前复核过程中出现的31语言/26综合临时计数已被最终标签覆盖。长度没有改善且身份一致性下降，不采用加权配方。

R38已启动两个学习率.003/.005各训练种子123/456。相同19969/MQA2/R4/1631原始不加权TRAIN/4k/B32/eval250/cosine5%warmup/data_seed42；连同旧seed42，每个配方三训练种子。四组40篇生成种子20261006/CPU/B8/T.5/top_p.9固定，最终100篇将使用新的生成种子，不把训练种子和生成种子混称。

### R38三训练种子结果与最终选择

六组均为19,969独立参数、V378/H32/I48、N1/R4、2查询头/1KV头、value residual .5、最终循环标准next-token CE；使用1631篇原始不加权TRAIN，Muon/B32/4k/eval250/cosine5%warmup/data_seed42。生成各40篇，固定CPU/B8/T.5/top_p.9/seed20261006，不强制EOS下限。

| 学习率 | 训练种子 | 恢复后验证CE | 最佳保存步 | 语言 | 闭合 | 身份一致 | 长度 | 四项同时通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| .003 | 42 | .874070 | 3750 | 33 | 38 | 37 | 33 | 25 |
| .003 | 123 | .874226 | 3750 | 33 | 40 | 39 | 29 | 24 |
| .003 | 456 | .869492 | 4000 | 34 | 40 | 38 | 35 | 29 |
| .005 | 42 | .862046 | 4000 | 31 | 40 | 38 | 36 | 27 |
| .005 | 123 | .863443 | 3750 | 30 | 38 | 35 | 33 | 24 |
| .005 | 456 | .860868 | 4000 | 33 | 39 | 33 | 37 | 28 |

.003三种子合计语言100/120、闭合118/120、身份114/120、长度97/120、综合78/120；.005分别94/117/106/106/79。综合只差1篇，低于单配方训练种子间24–29/40的波动；.003语言与身份连续性更好，因此选择其中综合29/40、验证CE最低的seed456。最终版本为`r38_20k_lr003_seed456`，没有用测试集选择权重。

### 冻结权重后的100篇与停止条件

冻结最终权重后，从BOS自主生成100篇，使用新生成种子20261007、CPU FP32/B8/T.5/top_p.9、min_new_tokens0/max_new_tokens2047。全部100篇正常EOS，0截断、0整篇重复，平均108.19词，平均重复4-gram率.030064。随机打乱后分成34/33/33篇，三代理看完整正文且不看候选元数据，按相同尺度复核：主线可读98、全文语言合格80、目标闭合98、身份连续91、100–300词82、四项同时通过65。原始样本与逐篇标签均保留；这些是AI辅助审阅，不是人类标注或开放域正确率。

最后两个学习率配方的综合差仅1/120，训练种子波动更大；R8综合22/40，未超过R35 R4的25/40；R37长稿重复权重综合27/40，但长度仍33/40、身份降到34/40；强制min_new_tokens的旧对照让长度合格40/40，却让语言和综合均降至16/40。新100篇综合65%与.003三训练种子平均65%一致。当前受控任务上继续这些调整的收益已经很小，因此结束本轮探索；这不构成所有约20k模型全局最优的证明。

最终导出在`model/`，已在工作区外的新进程通过AutoConfig/AutoTokenizer/AutoModelForCausalLM的`trust_remote_code=True`加载，独立参数19,969、输入输出词表权重共享；32 token前缀的完整/缓存推理FP32最大差1.90735e-5。最终测试集只在选择完成后评估一次：40篇、5678预测token，token加权CE .872573，2UNK，原始正文BPB为null。详细结果见`final_20k_report.md`与`hf_final_export_check.json`。

只保留最终导出权重：已从服务器工作区的`runs/`中移除171个旧权重/优化器文件，共47,238,556字节；配置、日志、指标、原始教师和学生样本及审阅记录保留。删除列表见`student_weight_cleanup.json`，服务器教师权重不在删除范围内。本地也只保留`model/model.safetensors`；交付压缩包包含同一最终版本。
