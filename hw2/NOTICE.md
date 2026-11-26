# 数据与第三方来源

本课程数据标识：`hw2-proteinnet7-cath43-20261103-v3`（2026-11-03）。

- ProteinNet：AlQuraishi Laboratory，CASP7 human-readable `training_50`。项目与格式：[官方仓库](https://github.com/aqlaboratory/proteinnet)、[记录格式](https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md)。仓库 MIT 许可全文保留于 `licenses/ProteinNet-MIT.txt`。原始公开结构来自 PDB，数据整理归功于 ProteinNet 作者。
- CATH 4.3.0：I. Sillitoe、N. Dawson、T. Lewis、D. Lee、J. Lees、C. Orengo 等，Protein Structure Classification Database。使用 `cath-domain-list-v4_3_0.txt` 的 homologous-superfamily 标注建立课程分组；[官方数据说明](https://www.cathdb.info/download)、[官方署名与许可声明](https://www.cathdb.info/wiki/doku/?id=start)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。课程对链、特征和分组进行了筛选与转换，不暗示原作者认可本课程结果。
- 下载 URL、原始文件散列、转换约定、链 ID、CATH 家族、划分及参考CNN预测来源见 `data/manifest.json` 与 `data/teacher_provenance.json`。

本数据是从 ProteinNet 的原训练数据构造的课程 train/validation/test，**不是官方 CASP7 测试基准**。完整链指保留源记录的全部序列位置；不表示所有坐标均已解析。只选48–96残基且至少85%坐标有效的链，保留缺失坐标的节点。数据代表性有限，CATH 分组也不等于穷尽所有远缘同源关系；不得把课程成绩外推为 AlphaFold 或 CASP 的精度。

正常流程由学生Part Two的plain CNN导出冻结预测。备用先验由同配置参考CNN生成，CNN仅用训练链更新参数，以验证集选检查点；未使用AlphaFold预训练权重或测试标签拟合模型。CNN与GNN复用训练链，训练先验属于in-sample预测，不宣称为折外预测。提供的TorchScript模型和构图器用于Part Four的完整GNN任务；跨任务比较所需的CNN训练/验证日志已提供，不分发CNN完整实现或已训练权重。
