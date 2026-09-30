# Exo→Ego推理信息契约

日期：2026-09-26

## 结论

当前H2O A–D实验全部属于 **Oracle上限实验**，不是“单路exo可部署推理”。特别是：

- 几何B使用H2O源相机的真值RGB-D、源相机位姿和目标cam4整段真值位姿；
- 状态C额外使用目标视角中的真值双手、物体6DoF和相机速度；
- 距离D额外使用由真值手关节和物体mesh计算的逐关节表面距离。

这些量可以用于回答“若物理状态正确，模型是否会利用它”，但不能证明它们能从exo可靠得到。

## 后续协议

| 协议 | 测试时允许的信息 | 研究问题 | 可否与真实ego逐像素比较 |
|---|---|---|---|
| O：Oracle upper bound | exo RGB-D、全部标定、真实ego轨迹、真实手物状态 | 各条件的收益上限与误差敏感性 | 可以 |
| A：Anchored practical | 单路exo RGB、人物track、一个ego首帧或初始相机位姿/FOV | 给定最小锚点后能否复现连续第一人称视角 | 可以，但须报告锚点 |
| C：Canonical ego | 单路exo RGB、人物track、用户指定或规范化FOV | 能否生成与动作一致的“合理第一人称”，不声称恢复真实注视 | 不应以单一GT逐像素分数为主 |
| M：Multi-exo | 多路同步exo RGB | 多视角是否改善动作状态与跨视角生成 | 可以 |
| MA：Multi-exo anchored | 多路同步exo RGB + 一个ego首帧 | 显式视角锚点与多视角动作证据能否解耦 | 可以 |

协议A是下一阶段推荐主线。没有ego首帧、初始朝向或用户控制时，单路exo通常不能唯一确定佩戴者的
头部光轴、眼睛注视、相机相对头骨安装偏差以及被遮挡区域；此时应转为协议C，输出带不确定性的
合理视角，而不是把一个猜测称为真实轨迹。

## 信息处理原则

1. **Oracle只做教师或审计。** 真实cam4位姿、cam4手物状态和接触距离不得进入可部署student的测试接口。
2. **可观测量才可硬约束。** exo RGB、人物框/2D关键点、由exo估计的深度和相机运动可以作为输入，
   但必须单独报告估计误差和下游敏感性。
3. **不可观测量作为分布。** 头部朝向、遮挡手势、物体背面和接触状态预测均输出置信度；低置信度时
   降低condition权重，由视频先验补全，不能伪装成精确物理约束。
4. **位置与视线分开。** 全局人体重建可提供头部中心的粗轨迹；相机光轴仍需ego首帧、可见面部/头部线索
   或用户控制。躯干朝向不能直接替代gaze。
5. **指标按协议解释。** O报告PSNR/深度一致性等上限；A同时报告相机旋转/平移误差和生成质量；
   C侧重动作、对象身份、手物关系、时序稳定和多样性，避免惩罚其他同样合理的视角。

## 当前模型的重新标注

- `rgb_only`：exo RGB基线；不含目标状态，但仍用同步ego作监督。
- `anchored_rgb`：exo RGB + 唯一的ego首帧；首帧显式声明为测试输入，不使用后续ego帧或轨迹。
- `anchored_residual`：相同输入，但以ego首帧为无损identity并只预测残差，防止有损重建先验。
- `multiview_rgb`：四路exo RGB；当前pilot为per-view全局特征均值，只是朴素下限。
- `multiview_anchored_residual`：四路exo + 唯一ego首帧的无损residual。
- `geometry`：Oracle geometry；当前不是可部署几何。
- `geometry_state`：Oracle geometry + Oracle state（去掉距离）。
- `geometry_state_distance`：Oracle geometry + Oracle state + Oracle contact distance。
- `rgb_state` / `state_only`：早期接口验证，不作为主实验结论。

目标位姿扰动实验已完成，3–5度旋转已显著破坏硬几何；`anchored_rgb`接口及首个同设置pilot也已完成。
静态复制ego首帧显著胜过原学习模型；无损anchor residual虽恢复画质，但运动幅度仅为真实值的0.11%。
四路exo朴素聚合也未改善变化区域。下一步用标定感知的空间融合/三角化提取手物动作状态，再做动态区域
运动残差加权，并将Oracle手物状态改成仅训练期可见的蒸馏目标。
