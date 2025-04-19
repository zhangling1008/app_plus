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
                    st.info("您的心理健康状况良好")
                elif all(r == 5 for r in responses):
                    cluster = 2  
                    st.info("建议寻求专业心理帮助")
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
                        "🎯 维持良好状态建议",
                        "" + "您的评估结果显示心理健康状态良好，这是值得珍惜的成果。为了长期维持这种状态，建议建立系统化的健康管理体系：首先，运动方面应保持每周3-5次、每次30-45分钟的有氧运动（如游泳、骑行或快走），这种规律运动能促进内啡肽分泌，提升压力耐受力。同时要重视睡眠质量，建议固定作息时间（22:30-6:30最佳），睡前1小时避免蓝光刺激，卧室温度保持在18-22℃之间。饮食上注意Omega-3脂肪酸的摄入（如深海鱼、坚果），这些营养素对神经细胞膜健康至关重要。建议每季度使用标准化量表（如PHQ-9、GAD-7）进行自评，建立个人心理健康档案，就像定期体检一样重要。",
                        
                        "" + "心理免疫力的建设需要持续投入：推荐每天进行10-15分钟的正念冥想训练，可使用专业APP如'Headspace'或'潮汐'引导，这种练习能增强前额叶对情绪调节的能力。同时建议每年参加4-6次心理健康讲座或工作坊，重点学习认知重构和情绪管理技巧。社交方面，保持每周至少3次有质量的面对面社交互动，这种真实连接能刺激催产素分泌，这是天然的抗焦虑物质。环境调节也很关键，可以在工作学习区域布置绿色植物或自然风景画，这类环境线索能降低压力激素水平。",
                        
                        "" + "要特别注意压力敏感期的预防性调节：在季节交替（特别是秋冬）、考试周、项目截止日前2周等时段，可以提前增加放松训练时长。推荐培养1-2个需要专注力的业余爱好（如绘画、乐器、园艺等），这类活动能产生心流体验，是天然的压力缓冲剂。当出现连续3天睡眠质量下降或情绪波动时，要及时启动'心理急救包'——可能是特定的音乐歌单、运动方案或信任朋友的联络方式。记住，心理健康的维护不是一劳永逸的，而是像保持身材一样需要持续的科学管理，您现在的良好状态正是系统养护的最佳起点。"
                    ],
                    1: [
                        "🛠️ 自助改善方案",
                        "" + "您的评估显示存在轻度心理困扰，这是可以通过系统方法有效改善的。首要任务是建立'情绪调节工具包'：强烈推荐'三件好事'日记法，每晚记录当天发生的三件积极事件及自己的贡献，这种练习能重塑大脑的消极偏好。同时掌握'4-7-8呼吸技术'（吸气4秒-屏息7秒-呼气8秒），每天晨起和睡前各练习5轮，这种有节奏的呼吸能直接调节自主神经系统。行为激活是关键，建议制定'S.M.A.R.T'微目标（如'今天散步15分钟并注意路边的三处美景'），小目标的达成会重建掌控感。睡眠改善方面，要建立严格的'数字戒断'制度——睡前90分钟避免所有电子设备，改用阅读或冥想替代，研究表明这能提升50%的睡眠效率。",
                        
                        "" + "认知重构训练需要循序渐进：当出现消极想法时，尝试'想法记录表'技术——写下触发事件、自动思维、情绪强度，然后寻找证据支持/反驳这个想法，最后形成更平衡的认知。推荐使用校心理中心提供的'心理自助资源包'，通常包含专业的放松指导音频、情绪调节手册和危机应对策略。营养方面要特别注意补充B族维生素（全谷物、深色蔬菜）和镁元素（坚果、香蕉），这些微量营养素是神经递质合成的必需物质。如果实施这些方法2周后，仍感觉疲惫感或消极情绪持续超过半天，建议预约专业心理咨询，早期干预的效果通常比问题严重后再处理要好3-5倍。",
                        
                        "" + "推荐精读马克·威廉姆斯的《改善情绪的正念疗法》，书中基于牛津大学研究的八周训练计划被证实能显著降低焦虑复发率。社交支持方面，建议识别3-5个'安全型'支持者（指那些能给予无条件积极关注的人），每周至少与他们进行1次有质量的交流。环境调整也很重要，可以重新布置生活空间——增加暖光源、减少杂乱物品、设置专门的放松角落。要特别注意'压力-休息'循环，每45-60分钟的高强度心理活动后，需要15分钟的完全放松（如冥想、拉伸）。这些方法看似简单，但系统性实施后，约70%的轻度心理困扰都能在6-8周内得到显著改善。"
                    ],
                    2: [
                        "⚡ 专业干预方案",
                        "" + "您的评估结果显示心理状态需要专业级别的关注和支持，请务必认真对待以下建议：安全计划是首要任务——立即确定三位24小时可联系的'紧急支持者'（建议包括1位专业人士如辅导员），将心理危机干预热线（如北京010-82951332）设置为手机快捷拨号。医疗干预方面，强烈建议72小时内采取行动：通过校医院精神科或当地三甲医院心理门诊进行专业评估，完整的诊断通常包括临床访谈、标准化量表和必要的生理检测。药物治疗方面，SSRIs类药物（如舍曲林、艾司西酞普兰）可能需要2-4周才能显效，期间务必遵医嘱定期复诊。同时要建立'安全环境清单'——暂时移除可能用于自我伤害的物品，避免独处在高楼、深水等高风险环境，这些预防措施能降低80%的急性风险。",
                        
                        "" + "心理治疗的选择需要个性化匹配：认知行为治疗（CBT）对抑郁和焦虑症状的改善率可达60-70%，通常需要12-20次系统治疗；接纳承诺治疗（ACT）则更适合反复出现的负面想法，其独特的关系框架技术能减少思维反刍。如果条件允许，建议同时参加支持性团体治疗，这种'共同体疗愈'能显著减轻病耻感。症状管理方面，必须坚持'三三制'基础自我照顾：每天保证3次规律进餐（即使少量）、3次补水（每次200ml）、3次5分钟的简单活动（如室内走动）。睡眠紊乱时可尝试'睡眠限制疗法'——严格限定卧床时间（如凌晨1点至6点），这种看似矛盾的方法反而能重建睡眠节律。药物治疗期间要特别注意酒精和咖啡因的完全戒断，这些物质会干扰药物代谢并加重症状波动。",
                        
                        "" + "康复过程需要系统支持：建议指定1位'治疗伙伴'（家人或密友）陪同参加重要诊疗，帮助记录医嘱和观察变化。在急性期（通常前4-8周），要暂缓做出重大人生决定（如休学、分手、辞职），因为此时认知功能可能受损20-30%。即使症状缓解后，仍需维持治疗3-6个月预防复发，就像抗生素需要完成整个疗程。可以创建'康复仪表盘'——每日记录关键指标（睡眠时长、情绪分数、药物依从性），这种可视化监测能提高30%的康复效率。要特别注意，心理困扰的康复往往呈螺旋式上升，期间可能出现2-3天的'症状反扑'，这完全是正常过程。请记住，寻求专业帮助不是软弱的表现，而是像骨折后就医一样的明智选择，现代精神医学对中重度心理困扰的有效缓解率已超过75%。"
                    ]
                }

                # 结果显示部分代码（使用HTML实现精确缩进控制）
                st.subheader("专业建议")
                for paragraph in cluster_advice.get(cluster, cluster_advice[2]):
                    if paragraph.startswith(("🎯", "🛠️", "⚡")):
                        st.markdown(f"## {paragraph}")
                    else:
                        st.markdown(f'<p style="text-indent: 2em; margin-bottom: 0.8em;">{paragraph}</p>', unsafe_allow_html=True)
                
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