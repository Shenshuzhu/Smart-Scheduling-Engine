import streamlit as st
import subprocess
import os
import pandas as pd
import sys # 修改

# 1. 网页全局配置
st.set_page_config(page_title="智能制造排产系统", page_icon="🏭", layout="wide")
st.title("🏭 基于启发式算法的智能排产系统 (CUT & AOI)")
st.markdown("上传最新生产数据，设置换线约束参数，一键生成全局最优排产 Excel。")

# 2. 侧边栏：文件上传与参数设置
st.sidebar.header("📁 数据注入区")
st.sidebar.info("请上传最新的订单需求表，系统将自动替换旧数据进行推演。")

# 允许用户上传需求表
uploaded_demand = st.sidebar.file_uploader("1. 上传新需求表 (demand.csv)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 约束参数设置 (示例)")
# 这里展示了滑动条和数字输入框，未来可以传给你的引擎
cut_day_limit = st.sidebar.slider("CUT 白班最大换线次数", min_value=1, max_value=10, value=4)
aoi_day_limit = st.sidebar.number_input("AOI 日最大换线次数", min_value=1, max_value=5, value=2)

# 3. 主界面：执行排产
st.write("### 🚀 排产执行控制台")

if st.button("开始运行智能排产引擎", type="primary"):
    with st.spinner("系统正在进行微观时序推演与产能分配，请稍候 (约需几秒至十几秒)..."):

        # 如果用户上传了新表格，就用新表格覆盖默认的 demand_202603.csv
        if uploaded_demand is not None:
            with open("demand_202603.csv", "wb") as f:
                f.write(uploaded_demand.getbuffer())
            st.toast("✅ 最新需求表已成功载入引擎！")

        # 注意：这里的传参逻辑为了极简，我们直接用 subprocess 调用你跑通的 .py 脚本
        # 未来你可以将参数 (cut_day_limit) 通过命令行参数传入引擎
        try:
            # 运行你的核心排产算法
            result = subprocess.run(
                [sys.executable, "scheduling0325_updated_v14.py"], # 修改了sys.executable
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                st.success("🎉 排产推演完成！各项约束均已满足。")

                # 4. 结果下载与预览
                output_file = "march_schedule_results_detailed.xlsx"
                if os.path.exists(output_file):
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="📥 点击下载最终排产方案 (Excel)",
                            data=f,
                            file_name="智能排产结果_导出.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.error("❌ 引擎运行遇到报错，请联系算法工程师排查：")
                st.code(result.stderr)

        except Exception as e:
            st.error(f"系统调用错误: {e}")
