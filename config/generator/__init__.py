
"""
地图与智能体生成逻辑。按功能分模块，外部按需从各子模块导入。

  - states_generator：起点、终点生成
  - forbidden_generator：障碍区生成
  - region_generator：LOS/NLOS 区域生成
  - environment_validation：环境验证
  - discretization：地图离散化与 O(1) state 查询
  - main：入口（调用上述模块生成 yml 并保存离散化与 O(1) 查询文件）
"""
