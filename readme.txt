haidian_price_pipeline_v3.py（高级版）

在v2版本基础上进行了全面增强，主要新增功能包括：
高级预测模型：
多模型候选与自动选择（线性回归、分段线性回归、Huber稳健回归等）
支持指数加权回归，更重视近期数据
自动分段线性回归，智能寻找趋势断点
预测增强：
阻尼外推技术（damped forecast）- 随时间降低预测斜率
预测区间计算与可视化
蒙特卡洛模拟生成多种可能的价格路径
回升概率计算（未来某个时间点价格高于当前价格的概率）
数据处理增强：
移动平均平滑减少月度噪声
价格下限保护机制
灵活的月度数据加载接口
丰富的命令行参数：支持20多种参数配置不同的预测场景
如何运行项目
1. 安装依赖
首先安装项目所需的Python包：

Bash



运行
pip install -r requirements.txt
2. 运行高级版本（v3）
方式一：使用模拟数据运行演示
这是最简单的方式，直接使用内置的模拟数据：
运行
python haidian_price_pipeline_v3.py --demo
方式二：使用月度均价数据
python haidian_price_pipeline_v3.py --monthly-file Data/haidian_monthly_prices_2015_2025.csv
方式三：使用自定义完整数据集
python haidian_price_pipeline_v3.py --file Data/haidian_house_prices_2015_2025_20251030_174444.csv
方式四：使用高级功能
# 使用预测区间和蒙特卡洛模拟
python haidian_price_pipeline_v3.py --demo --ci --mc-sims 1000

# 启用移动平均平滑，设置阻尼参数
python haidian_price_pipeline_v3.py --demo --smooth-ma 6 --phi 0.85

# 禁用分段线性模型，设置价格下限
python haidian_price_pipeline_v3.py --demo --no-piecewise --floor-pct 10

###  智能模型选择
v3版本会自动测试多种模型变体，包括：

- 全历史线性回归
- 不同窗口长度的滑动窗口线性回归（12/18/24/36个月）
- 不同半衰期的指数加权线性回归
- Huber稳健回归（对异常值更鲁棒）
- 分段线性回归（自动检测趋势变化点）
然后基于验证集性能自动选择最优模型。

### 2. 阻尼外推
考虑到长期预测中趋势可能不会持续，v3版本实现了阻尼外推功能，随着预测时间延长，趋势强度逐渐减弱，使长期预测更加合理。

### 3. 概率预测
通过蒙特卡洛模拟，v3版本可以生成多种可能的未来价格路径，并计算：

- 预测区间（不确定性范围）
- 价格回升概率
- 不同分位数的预测结果（10%/25%/50%/75%/90%）
### 4. 数据保护机制
- 移动平均平滑功能减少短期噪声
- 价格下限设置防止不合理的下跌预测