# GitHub 与 Zenodo 发布指南

你自行上传和发布。助手未创建仓库、Release 或 DOI。公开代码包已准备好；服务器实验包含私人清单和预测记录，不是 GitHub 上传包。

## 1 解压公开代码包

解压 Bioinspired_Attention_Evaluation_GitHub_20261013.zip。上传解压后的项目文件，而不是把整个 ZIP 作为仓库中唯一的文件。

项目包括 N3、原组件对照、新预算匹配控制、公共训练适配器、配对 bootstrap、Grad-CAM 和汇总效应量。模型配置侧重 N3 阶段；早期固定算子和设计空间的结果以汇总对照提供。精确复现历史实验仍需有权访问的数据、锁定清单、软件环境和审计记录。

## 2 填写真实元数据

将 CITATION.cff.example 中的作者姓、名、仓库地址、版本和许可证填写完整，随后重命名为 CITATION.cff。作者顺序与论文一致；不要保留示例占位符。

代码许可证由团队确认。确认后放入对应 LICENSE 文件，更新 CITATION.cff 的 SPDX 标识，删除 LICENSE_PENDING.md。数据、图片、预训练权重和第三方依赖各有其使用条件，代码许可证不能替代这些条件。

.zenodo.json.example 仅是参考模板。初次发布可以先使用填写完整的 CITATION.cff，再在 Zenodo 检查作者和许可证；不要同时发布相互矛盾的元数据。未分配的 DOI 不要自行填写。

九次预算匹配训练已完成。当前手稿共报告 96 次训练执行，其中包含 6 次相同种子的基线重跑，不能计为新增独立种子。公开包的 CPU 检查与已完成服务器实验应分别说明。

## 3 创建公开 GitHub 仓库

登录 [GitHub](https://github.com)，选择 New repository。建议仓库名为 bioinspired-attention-transfer-evaluation，选择 Public。

通过 Add file → Upload files 上传代码包中解压后的文件和目录。README.md、train_public.py、requirements.txt 和 CITATION.cff 应位于仓库根目录；src、configs、tests、vendor、docs 保持原目录结构。

提交后检查：

- README 正常显示，目录没有被额外包在两层同名文件夹里。
- 三个预算控制应标注为已完成训练；公开适配代码尚未在本地重复 GPU 训练，不得混淆两种验证范围。
- 数据图像、私人划分清单、逐图预测、模型二进制、令牌和密码不在仓库中。
- 模板作者、仓库地址和许可证占位符已经替换。

网页上传不会自动替你按 .gitignore 筛选文件；本次公开包已经排除了这些私人数据。不要把 stage31_budget_controls_v1_20261013.tar.gz 上传到这个公开仓库。

## 4 先连接 Zenodo，再创建 GitHub Release

登录 [Zenodo](https://zenodo.org)，使用 GitHub 授权连接，在 GitHub 集成设置中启用刚创建的公开仓库。组织仓库可能还需要组织允许访问。

确认仓库已启用后，回 GitHub 的 Releases 页面创建 v0.1.0，并选择正常发布。预发布标记不能替代正式归档流程。发布说明可写：

> Source code for the controlled evaluation of bioinspired attention in DR. Includes N3 and component controls, conditional paired image bootstrap, and completed attention-budget controls. The aggregate reports 27 comparisons from 96 training executions, including six baseline replays.

不要仅创建代码提交或标签就把它当作已获得 DOI 的归档版本。GitHub 与 Zenodo 的自动归档由发布流程触发，随后必须检查 Zenodo 中是否确实生成了对应软件记录。

## 5 核对 Zenodo 记录

核对软件记录的标题、作者顺序、所属机构、许可证、版本号、文件和描述。若记录尚未出现，不要在论文中填写一个猜测的 DOI；根据 Zenodo 的集成页面检查状态，必要时按其指引手动上传公开软件包。

Zenodo 通常提供特定版本 DOI 和涵盖所有版本的 concept DOI。论文复现材料建议引用实际采用的特定版本 DOI，同时可保留仓库地址。

## 6 将真实地址写入论文

发布成功后，Data Availability Statement 可以使用如下结构，替换所有占位符：

> The source code is available at [actual GitHub URL]. The software version used for this study is archived at [actual version-specific DOI]. Source datasets remain available from their providers under the applicable terms. Clinical images, private split manifests and checkpoint binaries are not redistributed in the public repository.

必须确保 “version used for this study” 与该版本实际包含的实现和状态一致。如果后续训练结果属于 v0.2.0，就不能用只有计划代码的 v0.1.0 冒充完整复现实验版本。

最后将 DOI 加入 CITATION.cff。下一版本的归档应继续使用真实版本号；不要为了修改说明而声称重新进行了旧实验。

## 官方说明

- [GitHub 引用仓库与 Zenodo DOI](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
- [Zenodo 启用 GitHub 仓库](https://help.zenodo.org/docs/github/enable-repository/)
- [Zenodo 归档 GitHub 软件版本](https://help.zenodo.org/docs/github/archive-software/github-upload/)

