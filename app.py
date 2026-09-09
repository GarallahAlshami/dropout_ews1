#!/usr/bin/env python3
"""نظام الإنذار المبكر للتسرب الدراسي — Streamlit production app."""
from pathlib import Path
import json
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "students.csv"
MODELS = BASE / "models"
METRICS = MODELS / "metrics.json"

st.set_page_config(page_title="نظام الإنذار المبكر للتسرب الدراسي", page_icon="🎓", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
html,body,[class*="css"]{font-family:'Tajawal',sans-serif!important;direction:rtl;text-align:right}
.stApp{background:linear-gradient(160deg,#f0f4f8,#e2e8f0)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e3a5f,#0f2744)}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:20px;box-shadow:0 4px 20px rgba(0,0,0,.06)}
.risk{font-size:2rem;font-weight:800}
</style>
""", unsafe_allow_html=True)

# The repository intentionally does not require committed binary model files.
# On first deployment, generate_and_train.py creates them deterministically.
def ensure_artifacts():
    required = [DATA, MODELS / "logistic_regression.joblib", MODELS / "xgboost.joblib", MODELS / "feature_cols.joblib", METRICS]
    if all(p.exists() for p in required):
        return
    with st.spinner("تهيئة البيانات وتدريب النماذج لأول تشغيل..."):
        result = subprocess.run([sys.executable, str(BASE / "generate_and_train.py")], cwd=str(BASE), capture_output=True, text=True)
        if result.returncode != 0:
            st.error("تعذر تجهيز النماذج.")
            st.code(result.stderr or result.stdout)
            st.stop()

ensure_artifacts()

@st.cache_data

def load_data():
    return pd.read_csv(DATA)

@st.cache_resource

def load_models():
    lr = joblib.load(MODELS / "logistic_regression.joblib")
    xgb = joblib.load(MODELS / "xgboost.joblib")
    features = joblib.load(MODELS / "feature_cols.joblib")
    with open(METRICS, encoding="utf-8") as f:
        metrics = json.load(f)
    return lr, xgb, features, metrics

df = load_data()
lr_model, xgb_model, feature_cols, metrics = load_models()
best_name = metrics["best_model"]
best_model = xgb_model if best_name == "xgboost" else lr_model

AR = {"gpa":"المعدل GPA","attendance":"الحضور","failed_courses":"المواد المتعثرة","assignment_rate":"تسليم الواجبات","participation":"المشاركة الأكاديمية","late_submissions":"التأخر في التسليم","system_activity":"النشاط في النظام","previous_warnings":"الإنذارات السابقة","absences":"الغيابات","credits_completed":"الساعات المكتملة","age":"العمر","financial_aid":"الدعم المالي","first_generation":"جيل أول","online_ratio":"نسبة الدراسة عن بعد"}

def risk(p):
    if p < .25: return "منخفض", "#059669"
    if p < .50: return "متوسط", "#d97706"
    if p < .75: return "مرتفع", "#dc2626"
    return "حرج", "#7c2d12"

def predict(features):
    X = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)
    p = float(best_model.predict_proba(X)[0,1])
    level, color = risk(p)
    return p, level, color

def recommendations(row):
    p=float(row.get("dropout_prob",0)); level,_=risk(p); out=[]
    if level=="حرج": out += [("عاجل","متابعة عاجلة من المرشد الأكاديمي خلال 48 ساعة."),("عاجل","وضع خطة تدخل فردية ومتابعة الأسرة عند الحاجة.")]
    elif level=="مرتفع": out.append(("تحذير","جدولة جلسة إرشاد أكاديمي وتدخل مبكر خلال أسبوع."))
    if float(row.get("attendance",100))<70: out.append(("تحذير","الحضور منخفض: تفعيل متابعة الحضور والتنبيه."))
    if float(row.get("gpa",4))<2: out.append(("تحذير","المعدل منخفض: إحالة للإرشاد وخطة رفع المعدل."))
    elif float(row.get("gpa",4))<2.5: out.append(("تنبيه","المعدل يحتاج تحسيناً: دعم أكاديمي أسبوعي."))
    if int(row.get("failed_courses",0))>=2: out.append(("تحذير","المواد المتعثرة: خطة دعم أو إعادة تسجيل مدروسة."))
    if float(row.get("assignment_rate",1))<.6: out.append(("تنبيه","ضعف تسليم الواجبات: متابعة أسبوعية."))
    if float(row.get("participation",1))<.4: out.append(("تنبيه","المشاركة الأكاديمية منخفضة: تشجيع المشاركة والأنشطة."))
    if int(row.get("late_submissions",0))>=5: out.append(("تنبيه","تأخر متكرر: دعم إدارة الوقت والمتابعة."))
    if int(row.get("previous_warnings",0))>=2: out.append(("تحذير","إنذارات سابقة متعددة: مراجعة شاملة للحالة."))
    return out or [("إيجابي","المؤشرات جيدة؛ الاستمرار في المتابعة الدورية.")]

st.sidebar.markdown("## 🎓 نظام الإنذار المبكر")
page=st.sidebar.radio("القائمة",["لوحة التحكم","تحليل طالب","الطلاب","مقارنة النماذج","أهمية المتغيرات","المحاكاة"],label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"النموذج الأفضل: {'الانحدار اللوجستي' if best_name=='logistic_regression' else 'XGBoost'}")
st.sidebar.caption(f"عدد الطلاب: {len(df):,}")

if page=="لوحة التحكم":
    st.title("لوحة التحكم")
    total=len(df); counts=df["risk_level"].value_counts(); avg=df["dropout_prob"].mean()*100
    cols=st.columns(5)
    for c,label,val in zip(cols,["إجمالي الطلاب","منخفض","متوسط","مرتفع","حرج"],[total,*[int(counts.get(x,0)) for x in ["منخفض","متوسط","مرتفع","حرج"]]]): c.metric(label,f"{val:,}")
    st.metric("متوسط احتمال التسرب",f"{avg:.1f}%")
    c1,c2=st.columns(2)
    with c1:
        rc=counts.reindex(["منخفض","متوسط","مرتفع","حرج"]).fillna(0)
        st.plotly_chart(px.pie(values=rc.values,names=rc.index,hole=.45,title="توزيع مستويات الخطر"),use_container_width=True)
    with c2:
        imp=metrics["feature_importance_xgb"]; top=dict(list(imp.items())[:8])
        st.plotly_chart(px.bar(x=list(top.values()),y=[AR.get(k,k) for k in top],orientation="h",title="أهم عوامل الخطر"),use_container_width=True)
    st.plotly_chart(px.scatter(df.sample(min(800,len(df)),random_state=42),x="gpa",y="attendance",color="risk_level",hover_data=["name","dropout_prob"],title="المعدل مقابل الحضور"),use_container_width=True)

elif page=="تحليل طالب":
    st.title("تحليل طالب جديد")
    with st.form("student"):
        a,b,c=st.columns(3)
        with a: gpa=st.slider("المعدل GPA",.5,4.,2.5,.05); attendance=st.slider("الحضور %",20.,100.,75.,.5); failed=st.number_input("المواد المتعثرة",0,10,1); assignment=st.slider("تسليم الواجبات",0.,1.,.7,.01)
        with b: participation=st.slider("المشاركة",0.,1.,.6,.01); late=st.number_input("مرات التأخر",0,30,3); activity=st.slider("النشاط في النظام",0.,1.,.5,.01); warnings=st.number_input("الإنذارات السابقة",0,10,1)
        with c: absences=st.number_input("الغيابات",0,60,8); credits=st.number_input("الساعات المكتملة",0,150,40); age=st.number_input("العمر",17,35,20); aid=st.selectbox("دعم مالي",[0,1],format_func=lambda x:"نعم" if x else "لا"); first=st.selectbox("جيل أول",[0,1],format_func=lambda x:"نعم" if x else "لا"); online=st.slider("الدراسة عن بعد",0.,1.,.3,.01)
        go_btn=st.form_submit_button("🔍 تحليل الطالب",use_container_width=True)
    if go_btn:
        feats={"gpa":gpa,"attendance":attendance,"failed_courses":failed,"assignment_rate":assignment,"participation":participation,"late_submissions":late,"system_activity":activity,"previous_warnings":warnings,"absences":absences,"credits_completed":credits,"age":age,"financial_aid":aid,"first_generation":first,"online_ratio":online}
        p,level,color=predict(feats); st.metric("احتمال التسرب",f"{p*100:.1f}%"); st.markdown(f"### مستوى الخطر: <span style='color:{color}'>{level}</span>",unsafe_allow_html=True)
        st.subheader("التوصيات")
        for kind,text in recommendations({**feats,"dropout_prob":p}): st.info(f"{kind}: {text}")

elif page=="الطلاب":
    st.title("قائمة الطلاب")
    q=st.text_input("بحث بالاسم أو الرقم"); rf=st.multiselect("مستوى الخطر",["منخفض","متوسط","مرتفع","حرج"]); min_gpa=st.slider("الحد الأدنى للمعدل",0.,4.,0.,.1)
    view=df.copy()
    if q: view=view[view["name"].str.contains(q,case=False,na=False)|view["student_id"].str.contains(q,case=False,na=False)]
    if rf: view=view[view["risk_level"].isin(rf)]
    view=view[view["gpa"]>=min_gpa]
    cols=["student_id","name","gpa","attendance","dropout_prob","risk_level","status"]
    show=view[cols].copy(); show["dropout_prob"]=(show["dropout_prob"]*100).round(1).astype(str)+"%"; show.columns=["رقم الطالب","الاسم","المعدل","الحضور %","احتمال التسرب","مستوى الخطر","الحالة"]
    st.dataframe(show,use_container_width=True,height=520)
    st.caption(f"عدد النتائج: {len(view):,}")

elif page=="مقارنة النماذج":
    st.title("مقارنة النماذج")
    lr=metrics["logistic_regression"]; xg=metrics["xgboost"]; names=["Accuracy","Precision","Recall","F1","ROC-AUC"]
    st.success(f"النموذج الأفضل حسب F1: {'Logistic Regression' if best_name=='logistic_regression' else 'XGBoost'}")
    fig=go.Figure([go.Bar(name="Logistic Regression",x=names,y=[lr[k.lower() if k!='F1' else 'f1'] for k in ["Accuracy","Precision","Recall","F1","ROC-AUC"]]),go.Bar(name="XGBoost",x=names,y=[xg[k.lower() if k!='F1' else 'f1'] for k in ["Accuracy","Precision","Recall","F1","ROC-AUC"]])]); fig.update_layout(barmode="group",yaxis_range=[0,1],title="أداء النموذجين"); st.plotly_chart(fig,use_container_width=True)
    st.json(metrics)

elif page=="أهمية المتغيرات":
    st.title("أهمية المتغيرات")
    imp=metrics["feature_importance_xgb"]; st.plotly_chart(px.bar(x=list(imp.values()),y=[AR.get(k,k) for k in imp],orientation="h",title="XGBoost — أهمية المتغيرات"),use_container_width=True)
    lrimp=metrics["feature_importance_lr"]; st.plotly_chart(px.bar(x=list(lrimp.values()),y=[AR.get(k,k) for k in lrimp],orientation="h",title="Logistic Regression — معاملات المتغيرات"),use_container_width=True)

else:
    st.title("المحاكاة")
    scenarios={"1. منخفض الخطورة":{"gpa":3.6,"attendance":94,"failed_courses":0,"assignment_rate":.95,"participation":.88,"late_submissions":0,"system_activity":.85,"previous_warnings":0,"absences":2,"credits_completed":70,"age":21,"financial_aid":1,"first_generation":0,"online_ratio":.2},"2. متوسط الخطورة":{"gpa":2.6,"attendance":78,"failed_courses":1,"assignment_rate":.72,"participation":.55,"late_submissions":3,"system_activity":.5,"previous_warnings":1,"absences":10,"credits_completed":45,"age":20,"financial_aid":0,"first_generation":1,"online_ratio":.4},"3. مرتفع الخطورة":{"gpa":1.9,"attendance":62,"failed_courses":3,"assignment_rate":.45,"participation":.3,"late_submissions":8,"system_activity":.25,"previous_warnings":2,"absences":22,"credits_completed":28,"age":22,"financial_aid":0,"first_generation":1,"online_ratio":.6},"4. حرج":{"gpa":1.2,"attendance":45,"failed_courses":5,"assignment_rate":.25,"participation":.15,"late_submissions":15,"system_activity":.1,"previous_warnings":4,"absences":35,"credits_completed":15,"age":23,"financial_aid":0,"first_generation":1,"online_ratio":.8},"5. حدّية":{"gpa":2.35,"attendance":71,"failed_courses":2,"assignment_rate":.58,"participation":.42,"late_submissions":5,"system_activity":.38,"previous_warnings":1,"absences":15,"credits_completed":38,"age":21,"financial_aid":1,"first_generation":0,"online_ratio":.35}}
    choice=st.radio("اختر حالة",list(scenarios));
    if st.button("تشغيل التنبؤ بالنموذج الحقيقي",use_container_width=True):
        p,level,color=predict(scenarios[choice]); st.metric("احتمال التسرب",f"{p*100:.1f}%"); st.markdown(f"### مستوى الخطر: <span style='color:{color}'>{level}</span>",unsafe_allow_html=True)
        for kind,text in recommendations({**scenarios[choice],"dropout_prob":p}): st.info(f"{kind}: {text}")

st.sidebar.markdown("---"); st.sidebar.caption("Dropout EWS v1.0 — بيانات محاكاة لأغراض المشروع الأكاديمي")
