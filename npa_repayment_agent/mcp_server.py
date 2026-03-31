from fastmcp import FastMCP

from npa_repayment_agent.pipeline import (
    build_collection_strategy_report,
    predict_repayment_probability,
    preprocess_npa_data,
    train_repayment_model,
)

mcp = FastMCP("npa-repayment-agent")


@mcp.tool
def preprocess_npa_data_tool(file_path: str, output_dir: str = "") -> dict:
    """预处理无抵押债务/NPA数据，生成清洗后的M/T数据集和数据画像。"""
    return preprocess_npa_data(file_path=file_path, output_dir=output_dir or None)


@mcp.tool
def train_repayment_model_tool(file_path: str, output_dir: str = "") -> dict:
    """训练无抵押债务3年付款预测模型，输出模型文件、指标和T集评分结果。"""
    return train_repayment_model(file_path=file_path, output_dir=output_dir or None)


@mcp.tool
def predict_repayment_probability_tool(file_path: str, model_path: str, output_dir: str = "") -> dict:
    """使用已训练模型为新组合打分，输出回款概率、EV代理值和催收分层。"""
    return predict_repayment_probability(file_path=file_path, model_path=model_path, output_dir=output_dir or None)


@mcp.tool
def build_collection_strategy_report_tool(file_path: str, output_dir: str = "") -> dict:
    """对指定Excel数据集完整执行预处理、训练、评估和催收策略报告生成。"""
    return build_collection_strategy_report(file_path=file_path, output_dir=output_dir or None)


if __name__ == "__main__":
    mcp.run()
