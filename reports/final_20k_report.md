# MacroStories 20k 最终研究与交付报告

连续探索至R38后，最终保留r38_20k_lr003_seed456，19,969个独立参数。三个训练种子与新的100篇生成复核已经完成。本轮试过的后续优化已进入收益较小的区间；这不是20k模型全局最优的证明。没有使用GRPO/DPO/RLHF，Gemma用于合成数据，正文质量由AI代理全文复核。

## 最终结构与训练

WordLevel V378；hidden size32；SwiGLU intermediate48；一个物理Transformer层共享执行四次；2查询头、1KV头，head_dim16；RoPE；attention/FFN分支前后RMSNorm；每轮共用输出RMSNorm；value residual mix .5。首次attention的V是后续各遍V混合的锚。最终循环的标准next-token CE，无auxiliary loop loss、循环向量或输入再注入。

参数：共享embedding/output12,096；attention3,072；SwiGLU4,608；RMSNorm160；原Ouro保留但未用于提前退出的gate33；合计19,969。四次循环增加计算而不复制权重。源代码来自固定版本ByteDance/Ouro，使用HF Auto类导出及custom code加载，从随机权重训练，没有载入Ouro大模型权重。

Muon更新decoder二维Linear矩阵，其余权重AdamW；LR .003，weight decay .01，B32，4,000更新，每250步验证/保存，cosine、5%warmup、clip1，训练seed456、data_seed42。最佳保存step4000；Trainer恢复验证CE .869492。最终配置见experiments/final_20k.json。

## 数据实际规模

最终TRAIN1,631篇唯一完整原稿，233,756tokens含BOS/EOS；validation/test各40；词表在初始328篇TRAIN上构造并冻结V378。当前TRAIN零UNK，validation3UNK、test2UNK，均为未见表面词形；不将原始正文BPB报告成有效指标。新语料只进入TRAIN，冻结heldout不参与词表更新。

| V4数据批次 | 请求 | 格式接受 | 全文语言/闭合复核通过 | 加入当前数据 |
|---|---:|---:|---:|---:|
| regex pilot | 96 | 62 | 57 | 与stage1合并 |
| stage1 | 516 | 386 | 351 | 合计408，再切328/40/40 |
| stage2 | 1032 | 734 | 688 | 去掉39篇冻结词表UNK，新增649TRAIN |
| stage3 | 1032 | 768 | 690（三项含身份AND） | 去掉36篇冻结词表UNK，新增654TRAIN |

stage2另有293格式拒稿、5HTTP超时；stage3有264格式拒稿、0API错误；未对失败请求自动重试。全部原稿/拒稿与审阅保留在my-gpu-server。stage3语言693/768、闭合768/768、身份765/768、三项AND690/768；增加身份筛选，不能声称只是数据大小改变。V4流程四批合计2676请求、1950格式接受，最后1631TRAIN+80heldout=1711篇。六类TRAIN计数：带面包206、建鸟屋210、修风筝271、取红球324、还书270、浇花350。

教师使用已有Gemma4 26B A4B W4A16/vLLM，生成T1.0/top_p.95/top_k64、思考预算16。教师词形regex约束有限英文词形与人物标记，学生生成直接使用HF generate。显示函数仅将人物编号映射为Alex/Robin/Casey。

## 训练种子复验

同一V378/1631TRAIN/4k/R4/MQA2、CPU/B8/T.5/p.9，筛选生成seed20261006。language_clean要求全文无明确语言错误；closure要求开头目标、相关解决行动及同目标结果；identity检查人物、物件和重要状态；combined为这三项及100–300词的AND。readable允许局部错误，另列。

| LR | 训练seed | 验证CE | language | closure | identity | length | combined/40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 42 | 0.874070 | 33 | 38 | 37 | 33 | 25 |
| 0.003 | 123 | 0.874226 | 33 | 40 | 39 | 29 | 24 |
| 0.003 | 456 | 0.869492 | 34 | 40 | 38 | 35 | 29 |
| 0.005 | 42 | 0.862046 | 31 | 40 | 38 | 36 | 27 |
| 0.005 | 123 | 0.863443 | 30 | 38 | 35 | 33 | 24 |
| 0.005 | 456 | 0.860868 | 33 | 39 | 33 | 37 | 28 |

.003配方三种子累计language100/120、closure118/120、identity114/120、length97/120、combined78/120；.005分别94/120、117/120、106/120、106/120、79/120。综合差仅1/120，远小于种子间24–29/40的波动。.003的语言和连续性更好，因此选它；seed456同时为该配方最低验证CE、最高本组语言/综合计数。筛选用40篇不能视为独立最终正确率。

后续R8综合22/40，原R4为25/40；长度加权1994记录/1631unique最终33语言、34身份、33长度、27综合，长度未提高；HF min_new_tokens152让旧long8k长度达标40/40，却使语言31→16、综合24→16，重复增多。没有保留这些改动。共享嵌入梯度放大、n-gram记忆等前沿方向只有大规模或不一致收益证据，研究记录了适用边界，未实施或宣称复现。当前不再添加缺乏正向证据的组合。

## 冻结权重后的100篇

生成seed20261007，CPU/B8，T.5/top_p.9，BOS-only、max2047、min0。三个AI子代理只读随机混排、隐藏模型元数据的34/33/33篇正文，保留原sample_index，全文独立标注；没有使用Gemma评分当标签。100条标签覆盖100个不同ID。质量计数属于AI辅助评价，不是人工标注或客观成功率。

| 项目 | 通过/100 |
|---|---:|
| readable | 98 |
| language_clean | 80 |
| closure | 98 |
| identity_consistent | 91 |
| length_ok | 82 |
| combined | 65 |

100/100正常EOS、0截断、0整篇重复；平均108.19词，平均重复4-gram rate .030064。18篇短于100词。即使故事结束且主线可读，也可能有局部语言错误或人物／物体关系错误；65篇同时通过四项严格标准。此独立计数与.003三训练种子的综合均值65%一致，支持当前配方稳定在相近区间，不能外推到任意题材或词汇。

原始正文final100_samples_t05.jsonl、汇总标签final100_labels.jsonl；分段判断及证据在final100_part1/2/3_review.md。成功示例是原sample20，106词、全部条件通过，未经改写：

> "I want to help this flower," Alex said. The flower was dry in the village garden. Alex wanted to water the flower. Alex grabbed a large bucket, but the bucket was too heavy for them to lift. Alex looked at the small cup on the bench. Alex dipped the cup into the bucket. Alex walked to the flower and poured the water onto the soil. Alex went back and forth. Alex made several trips with the small cup. Alex poured more water into the dirt. Alex and Robin watched the plant. Finally, the flower got enough water. Alex smiled because the flower was not dry now.

失败原sample11出现“The flower got enough water and poured Alex.”；sample37出现“Alex had a good and held the bread inside the blue sky.”，均保留完整原文，显示剩余施事、宾语和句法混接问题。

## HF及测试集验证

在服务器/tmp新进程，AutoConfig/AutoTokenizer/AutoModelForCausalLM(trust_remote_code=True)成功加载，只依赖导出目录中的custom code。参数19,969，input/output共享同一存储；32-token完整/增量cache logits最大绝对差1.90735e-5（FP32），没有声称逐bit相同。导出权重文件81,364bytes，配置与源码一并提供。

最后仅一次评估未用于配方选择的40篇test：CPU FP32、token加权CE .872573，预测5678tokens含EOS、不预测BOS，2UNK；因此原始文本BPB为null。该指标的加权方式与Trainer验证CE不同，不能仅据数值差作训练过拟合结论。验证记录hf_final_export_check.json、final_test_likelihood.json。

## 交付范围

model/为唯一保留版本，支持HF库custom code加载；MacroStories-20k.zip打包同一版本。历史实验配置、训练日志、原始输出和报告保留，旧候选权重移除。模型仅在六种受控儿童情节及有限词表范围验证，BOS自主生成；没有验证任意开放主题提示续写。原始样本可直接核对语言、连续性与重复，不能把65/100解释为一般写作成功率。

交付目录包含10个文件，共148,756字节，其中safetensors权重81,364字节；压缩包94,493字节，包含同一目录的权重、配置、tokenizer、custom code、姓名显示函数及模型说明。本地和my-gpu-server均保存该压缩包。服务器旧实验权重及优化器文件共移除171个、47,238,556字节，删除清单见student_weight_cleanup.json。
