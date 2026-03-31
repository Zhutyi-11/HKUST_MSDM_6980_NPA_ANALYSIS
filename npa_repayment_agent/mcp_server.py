from fastmcp import FastMCP

from npa_repayment_agent.pipeline import (
    build_collection_strategy_report,
    optimize_collection_policy,
    predict_repayment_probability,
    preprocess_npa_data,
    train_repayment_model,
)

mcp = FastMCP("npa-repayment-agent")


@mcp.tool
def preprocess_npa_data_tool(file_path: str, output_dir: str = "", config_path: str = "") -> dict:
    """预处理无抵押债务/NPA数据，生成清洗后的M/T数据集、数据画像和生产配置快照。"""
    return preprocess_npa_data(file_path=file_path, output_dir=output_dir or None, config_path=config_path or None)


@mcp.tool
def train_repayment_model_tool(file_path: str, output_dir: str = "", config_path: str = "") -> dict:
    """训练投产版无抵押债务3年付款预测模型，输出校准模型、Champion-Challenger结果和生产队列汇总。"""
    return train_repayment_model(file_path=file_path, output_dir=output_dir or None, config_path=config_path or None)


@mcp.tool
def predict_repayment_probability_tool(file_path: str, model_path: str, output_dir: str = "", config_path: str = "") -> dict:
    """使用已训练生产模型为新组合打分，输出校准后的回款概率、队列建议和预期净回收。"""
    return predict_repayment_probability(
        file_path=file_path,
        model_path=model_path,
        output_dir=output_dir or None,
        config_path=config_path or None,
    )


@mcp.tool
def optimize_collection_policy_tool(file_path: str, model_path: str, output_dir: str = "", config_path: str = "") -> dict:
    """基于已训练模型和经济假设优化催收策略，输出推荐队列、净回收汇总和配置快照。"""
    return optimize_collection_policy(
        file_path=file_path,
        model_path=model_path,
        output_dir=output_dir or None,
        config_path=config_path or None,
    )


@mcp.tool
def build_collection_strategy_report_tool(file_path: str, output_dir: str = "", config_path: str = "") -> dict:
    """对指定Excel数据集执行投产版完整流程：预处理、校准建模、策略优化和管理层报告生成。"""
    return build_collection_strategy_report(file_path=file_path, output_dir=output_dir or None, config_path=config_path or None)


if __name__ == "__main__":
    mcp.run()
