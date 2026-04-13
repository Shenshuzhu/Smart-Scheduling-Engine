import streamlit as st
import subprocess
import os
import pandas as pd
import sys # 调用sys.executable

# 1. 网页全局配置
st.set_page_config(page_title="智能制造排产系统", page_icon="🏭", layout="wide")
st.title("🏭 基于启发式算法的智能排产系统 (CUT & AOI)")
st.markdown("上传生产数据与约束规则，一键生成排产 Excel。")

st.markdown("---")

# 2. 主界面：多源数据注入
st.subheader("📁 第一步：生产数据与规则注入")
st.info("💡 提示：按需上传对应的 CSV 文件即可替换系统配置，未上传的项将自动沿用服务器上已有的默认数据。")

col1, col2 = st.columns(2) # 使用列布局，上传框更美观

# 用户上传
with col1:
    st.markdown("#### 📦 订单与初始状态")
    up_demand = st.file_uploader("1. 需求表 (demand_202603.csv)", type=['csv'])
    up_hist = st.file_uploader("2. 初始机器状态 (hist_260313.csv)", type=['csv'])
    up_hist_recipe = st.file_uploader("3. 历史配方表 (hist_recipe.csv)", type=['csv'])

with col2:
    st.markdown("#### ⚙️ 设备能力与换线规则")
    up_cut = st.file_uploader("4. CUT 产能表 (cut_cap.csv)", type=['csv'])
    up_aoi = st.file_uploader("5. AOI 产能表 (aoi_cap.csv)", type=['csv'])
    up_changeover = st.file_uploader("6. 换线时间矩阵 (changeover.csv)", type=['csv'])
    up_irregular = st.file_uploader("7. 异形产品名单 (irregular.csv)", type=['csv'])

st.markdown("---")

# 3. 主界面：执行排产
st.subheader("🚀 第二步：排产推演")

if st.button("开始运行智能排产引擎", type="primary"):
    with st.spinner("系统正在进行时序推演与产能分配，请稍候 (约需几秒至十几秒)..."):
        
        # 将用户上传的文件与底层要求的文件名进行映射绑定
        upload_mapping = {
            "demand_202603.csv": up_demand,
            "hist_260313.csv": up_hist,
            "hist_recipe.csv": up_hist_recipe,
            "cut_cap.csv": up_cut,
            "aoi_cap.csv": up_aoi,
            "changeover.csv": up_changeover,
            "irregular.csv": up_irregular
        }
        
        # 遍历字典，如果哪个文件被上传了，就覆盖掉服务器上的旧文件
        replaced_count = 0
        for filename, uploaded_file in upload_mapping.items():
            if uploaded_file is not None:
                with open(filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                replaced_count += 1
                
        if replaced_count > 0:
            st.toast(f"✅ 成功载入 {replaced_count} 份最新业务数据表！")
            
        try:
            # 运行核心排产算法
            result = subprocess.run(
                [sys.executable, "scheduling0325_updated_v14.py"], # 修改调用python为sys.executable
                capture_output=True,
                text=True,
                encoding="utf-8" # 强制指定编码为Linux默认编码UTF-8
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
                st.error("❌ 引擎运行遇到错误，请检查上传的表格格式是否正确，或联系工程师排查：")
                st.code(result.stderr)

        except Exception as e:
            st.error(f"系统调用错误: {e}")
