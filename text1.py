import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import qrcode
from io import BytesIO
import base64
import socket
import sqlite3
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 1. 初始化数据库
def init_db():
    conn = sqlite3.connect('scl90_data.db')
    c = conn.cursor()
    
    # 创建问卷结果表
    c.execute('''CREATE TABLE IF NOT EXISTS responses 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT NOT NULL,
                  timestamp DATETIME,
                  responses TEXT,
                  cluster INTEGER,
                  factor_scores TEXT)''')
    
    # 尝试添加列（如果表已存在但缺少该列）
    try:
        c.execute("ALTER TABLE responses ADD COLUMN student_id TEXT NOT NULL DEFAULT 'unknown'")
    except sqlite3.OperationalError as e:
        # 列已存在时会报错，可以忽略
        if "duplicate column name" not in str(e):
            raise
    
    # 删除不再需要的用户信息表
    c.execute("DROP TABLE IF EXISTS user_info")
    
    conn.commit()
    conn.close()
# 初始化数据库
init_db()

# 2. 加载模型和数据
@st.cache_resource
def load_models():
    scaler = StandardScaler()
    autoencoder = load_model('scl90_autoencoder.h5')
    embeddings = np.load('embeddings.npy')
    
    # 加载标准化参数
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.scale_ = np.load('scaler_scale.npy')
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(embeddings)
    return autoencoder, scaler, kmeans

autoencoder, scaler, kmeans = load_models()


# 4. 获取本地IP地址
def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except:
        return "localhost"

# 5. 完整的SCL-90问卷题目
questions = [
    "1. 头痛", "2. 神经过敏，心中不踏实", "3. 头脑中有不必要的想法或字句盘旋", "4. 头昏或昏倒", "5. 对异性兴趣减退",
    "6. 对旁人责备求全", "7. 感到别人能控制您的思想", "8. 责怪他人制造麻烦", "9. 忘记性大", 
    "10. 担心自己的衣饰整齐以及仪态端正", "11. 易烦恼激动", "12. 胸痛", "13. 害怕空旷场所或街道",
    "14. 感到自己精力下降，活动减慢", "15. 想结束自己的生命", "16. 听到旁人听不到的声音", "17. 发抖", "18. 感到大多数人都不可信任",
    "19. 胃口差", "20. 容易哭泣", "21. 与异性相处害羞不自在", "22. 感到受骗、中了圈套或有人想抓住您", 
    "23. 无缘无故地感到害怕", "24.自己不能控制地大发脾气", "25. 怕单独出门", "26. 经常责怪自己",
    "27. 腰痛", "28. 感到难以完成任务", "29. 感到孤独", "30. 感到苦闷", "31. 过分担忧",
    "32. 对事物不感兴趣", "33. 感到害怕", "34. 您的感情易受伤害", "35. 旁人知道自己的私下想法",
    "36. 感到别人不理解您不同情您","37. 感到人们对您不友好，不喜欢您", "38. 做事必须做得很慢以保证做得正确", "39. 心跳得很厉害",
    "40. 恶心或胃部不适", "41. 感到比不上他人", "42. 肌肉酸痛", "43.感觉有人在监视您谈论你",
    "44. 入睡困难", "45. 做事必须反复检查", "46. 难以作出决定", "47. 怕乘电车、公共汽车、地铁或火车",
    "48. 呼吸困难", "49. 一阵阵发冷或发热", "50.因为感到害怕而避开某些东西、场合或活动", "51. 脑子变空",
    "52. 身体发麻或刺痛", "53. 喉咙有梗塞感", "54. 感觉前途没有希望", "55. 不能集中注意",
    "56. 感到身体的某一部分软弱无力", "57. 感到紧张或容易紧张", "58. 感到手或脚发重", "59. 想到死亡的事情",
    "60. 吃得太多", "61. 当别人看着您或谈论您时感到不自在", "62. 有一些不属于您自己的想法", 
    "63. 有想打人或伤害他人冲动", "64. 醒得太早", "65. 必须反复洗手，点数目或触摸某些东西", "66. 睡得不稳不醒",
    "67. 有想摔坏或破坏东西的冲动", "68. 有一些别人没有的想法或念头", "69. 感到对别人神经过敏", "70. 在商店或电影院等人多的地方感到不自在",
    "71. 感到任何事情都很困难", "72. 一阵阵恐惧或惊慌", "73. 感到在公共场合吃东西很不舒服食不适", "74. 经常与人争论",
    "75. 单独一人时神经很紧张", "76. 别人对您的成绩没有做出恰当的评价", "77. 即使和别人在一起也感到孤独", "78. 感到坐立不安心神不定",
    "79. 感到自己没有什么价值", "80. 感到熟悉的东西变得陌生或不像是真的", "81. 大叫或摔东西", "82. 害怕会在公共场合昏倒",
    "83. 感到别人想占您的便宜", "84. 为一些有关“性”的想法而很苦恼", "85. 您认为应该因为自己的过错而受到惩罚", "86. 感到要赶快把事情做完",
    "87. 感觉身体有严重问题", "88. 从未感到和其他人很亲近", "89. 感到自己有罪", "90. 感觉自己脑子有毛病"
]

# 6. 保存数据到数据库
def save_to_db(student_id, responses, cluster, factor_scores):
    conn = sqlite3.connect('scl90_data.db')
    c = conn.cursor()
    
    # 保存问卷结果
    c.execute('''INSERT INTO responses 
                 (student_id, timestamp, responses, cluster, factor_scores) 
                 VALUES (?, ?, ?, ?, ?)''',
              (student_id,
               datetime.now(), 
               str(responses), 
               int(cluster), 
               str(factor_scores)))
    
    conn.commit()
    conn.close()
# 7. 创建问卷界面
st.title("SCL-90心理健康自评量表")

with st.sidebar:
    # 管理员登录（优化版）
    st.subheader("管理员登录")
    admin_pass = st.text_input("密码", type="password", key="admin_pass")
    
    # 添加明确的登录按钮
    login_button = st.button("登录")
    logout_button = st.button("注销") if 'admin' in st.session_state else None
    
    if login_button:
        if admin_pass == "admin123":  # 简单密码，实际使用中应该更安全
            st.session_state.admin = True
            st.success("管理员模式已激活")
        else:
            st.error("密码错误")
    
    if logout_button:
        del st.session_state.admin
        st.success("已退出管理员模式")

# 主问卷区域
with st.form("scl90_form"):
    # 学号输入
    student_id = st.text_input("学号*", help="请输入您的学号", key="student_id")
    
    st.subheader("请根据最近一周的感觉评分（1-5分）：")
    st.caption("1=没有，2=很轻，3=中等，4=偏重，5=严重")
    
    responses = []
    cols = st.columns(5)  # 分5列显示
    for i, q in enumerate(questions):
        with cols[i % 5]:
            responses.append(
                st.radio(
                    q,
                    options=[1, 2, 3, 4, 5],
                    horizontal=True,
                    key=f"q{i}"
                )
            )
    
    submitted = st.form_submit_button("提交评估")
    
    if submitted:
        if not student_id:
            st.error("请输入学号")
        elif len(responses) != 90:
            st.error("请确保回答了所有90个问题")
        else:
            try:
                # 计算因子得分
                factors = {
                    '躯体化': [0,3,11,26,39,41,47,48,51,52,55,57],
                    '强迫症状': [2,8,9,27,37,44,45,50,54,64],
                    '人际关系敏感': [5,20,33,35,36,40,60,68,72],
                    '抑郁': [4,13,14,19,21,25,28,29,30,31,53,70,78],
                    '焦虑': [1,16,22,32,38,56,71,77,79,85],
                    '敌对': [10,23,62,66,73,80],
                    '恐怖': [12,24,46,49,69,74,81],
                    '偏执': [7,17,42,67,75,82],
                    '精神病性': [6,15,34,61,76,83,84,86,87,89],
                    '其他': [18,43,58,59,63,65,88]
                }
                
                factor_scores = []
                for factor, indices in factors.items():
                    score = np.mean([responses[i] for i in indices])
                    factor_scores.append(score)
                
                
                if all(r == 1 for r in responses):
                    cluster = 0 
                    st.info("检测到全1输入，自动分类为健康状态")
                elif all(r == 5 for r in responses):
                    cluster = 2  
                    st.info("检测到全5输入，自动分类为严重状态")
                else:
                    # 原有分类逻辑
                    scaled = scaler.transform(np.array(factor_scores).reshape(1, -1))
                    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
                    embedding = encoder.predict(scaled, verbose=0)
                    cluster = kmeans.predict(embedding)[0]
                
                # 保存到数据库和后续显示逻辑...
                save_to_db(student_id, responses, cluster, factor_scores)
                st.success("评估完成！")
            
                st.info(f"您的学号: {student_id} (请妥善保存)")
                
                descriptions = {
                    0: "您的心理健康状况良好",
                    1: "存在轻度心理困扰",
                    2: "建议寻求专业心理帮助"
                }
                
                st.write(f"**评估结果**: {descriptions.get(cluster, '未知状态')}")
                
                st.subheader("因子得分雷达图")
                # 准备雷达图数据
                categories = list(factors.keys())
                values = factor_scores
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='因子得分'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]  # SCL-90评分范围是1-5
                        )),
                    showlegend=True,
                    title="SCL-90各因子得分雷达图"
                )
                
                st.plotly_chart(fig)
                

                # 显示因子得分表格
                st.subheader("各因子得分详情")

                # 创建带有HTML样式的表格
                factor_data = {
                    "因子名称": categories,
                    "平均得分": [f"{score:.2f}" for score in values],
                    "解释": [
                        "身体不适感" if score > 2.5 else "正常" if score < 1.5 else "轻微不适"
                        for score in values
                    ]
                }

                # 转换为DataFrame
                df_factors = pd.DataFrame(factor_data)

                # 定义颜色函数
                def color_values(val):
                    try:
                        score = float(val)
                        if score > 2.5:  # 严重情况
                            return 'color: red; font-weight: bold'
                        elif score >= 1.5:  # 轻度不适
                            return 'color: #FFA500; font-weight: bold'  # 橙色
                    except:
                        pass
                    return ''

                # 应用样式
                styled_df = df_factors.style.applymap(color_values, subset=['平均得分'])

                # 显示表格
                st.dataframe(styled_df)
                                
               
                st.subheader("个性化建议")
                
                cluster_advice = {
                            0: [
                                "🎯 维持良好状态建议：",
                                "• 继续保持当前健康的生活方式模式，每周保持3-5次、每次30分钟的中等强度运动",
                                "• 建议每3-6个月进行一次心理健康自评，可使用PHQ-9或GAD-7等简易量表",
                                "• 建立规律作息（建议睡眠时间22:30-6:30），保证7-8小时优质睡眠",
                                "• 可尝试正念冥想等预防性心理训练，推荐使用'Headspace'或'潮汐'等APP",
                                "",
                                "📈 提升建议：",
                                "• 参加心理健康知识讲座，提升心理韧性",
                                "• 培养1-2个减压爱好（如绘画、园艺等）"
                            ],
                            1: [
                                "⚠️ 重点关注领域：",
                                "• 您的评估显示在[抑郁/焦虑]维度存在轻度困扰",
                                "• 这些症状可能表现为：情绪波动增大、睡眠质量下降、注意力难以集中等情况",
                                "",
                                "🛠️ 自助改善方案：",
                                "• 情绪管理：每天记录'三件好事'日记，培养积极认知模式",
                                "• 放松训练：每天2次'4-7-8呼吸法'（吸气4秒-屏息7秒-呼气8秒）",
                                "• 行为激活：制定每日小目标（如散步15分钟、联系1位朋友）",
                                "• 睡眠改善：建立睡前1小时'数字戒断'习惯，保持卧室温度18-22℃",
                                "",
                                "👨‍⚕️ 专业支持建议：",
                                "• 如果上述症状持续2周以上未见改善，建议预约心理咨询",
                                "• 可先使用校心理中心提供的'心理自助资源包'（包含放松指导音频等）",
                                "• 推荐阅读：《改善情绪的正念疗法》（Williams著）"
                            ],
                            2: [
                                "🚨 重要提示：",
                                "• 您的评估结果显示多个维度（抑郁、焦虑等）显著高于常模水平",
                                "• 这些症状可能已经影响到您的学习效率、社交功能和日常生活质量",
                                "",
                                "⚡ 立即行动建议：",
                                "1. 安全计划：",
                                "   - 识别3个可以随时联系的支持人员（亲友/辅导员等）",
                                "   - 保存心理危机干预热线（北京心理危机干预中心010-82951332）",
                                "2. 专业帮助：",
                                "   - 建议3天内预约心理咨询师进行专业评估",
                                "   - 可联系校医院精神科或当地专科医院心理门诊",
                                "3. 自我照顾：",
                                "   - 避免独处于高风险环境（如高处、危险物品附近）",
                                "   - 保持基础生活节律（至少保证1日3餐、4小时睡眠）",
                                "",
                                "🏥 治疗选择参考：",
                                "• 心理咨询：认知行为治疗（CBT）或接纳承诺治疗（ACT）",
                                "• 必要时可考虑药物辅助治疗（需专业医生评估）",
                                "• 团体治疗：校心理中心可能提供相关支持小组",
                                "",
                                "📌 注意事项：",
                                "• 避免使用酒精或不规范药物自我治疗",
                                "• 暂时减少重大决定（如休学、分手等）",
                                "• 症状缓解后仍需维持4-8周的巩固期"
                            ]
                        }
                for advice in cluster_advice.get(cluster, cluster_advice[2]):
                    st.markdown(f"- {advice}")
                
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
                st.info("请确保已回答所有问题并重新提交")

# 管理员数据查看
if 'admin' in st.session_state and st.session_state.admin:
    conn = sqlite3.connect('scl90_data.db')
    df = pd.read_sql('SELECT * FROM responses', conn)
    conn.close()
    
    st.subheader("所有问卷数据")
    st.dataframe(df)
    
    # 添加数据分析功能
    st.subheader("数据分析")
    if not df.empty:
        st.write(f"总收集量: {len(df)}")
        st.bar_chart(df['cluster'].value_counts())
        
        # 导出数据
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "导出数据为CSV",
            data=csv,
            file_name='scl90_responses.csv',
            mime='text/csv'
        )