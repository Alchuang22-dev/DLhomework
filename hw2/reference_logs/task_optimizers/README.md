# CNN优化器参考日志

三份真实CPU记录，任务为Part Two的plain CNN，seed416，各200更新，相同初值、窗口和裁剪。学习率SGD0.05、Momentum0.01、Adam0.001；Momentum0.9，Adam betas=(0.9,0.999)、epsilon=1e-8。每50步用完整验证NLL选点。本目录只提供训练/验证字段，不提供测试指标。

Part Four必做实验使用Part Three的predicted-edge GNN，Adam学习率为1e-4。比较两项任务时同时考虑任务、学习率配置和有限预算，不把顺序变化全部归因为模型结构。不要求学生重新运行本目录的CNN实验。
