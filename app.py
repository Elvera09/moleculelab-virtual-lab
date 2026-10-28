import streamlit as st
import pandas as pd
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO

st.set_page_config(page_title="MoleculeLab | Molecular Intelligence Studio", page_icon="🧬", layout="wide")

# Original glassmorphism-inspired interface
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --ink:#101d5c; --muted:#53699e; --violet:#7148ff; --blue:#38a5ff; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp {
 background: radial-gradient(circle at 15% 10%, #f9eaff 0, transparent 30%),
             radial-gradient(circle at 85% 15%, #c8e8ff 0, transparent 35%),
             linear-gradient(135deg,#edf0ff 0%,#d9eaff 50%,#eee5ff 100%);
 color:var(--ink);
}
.block-container { padding-top:1.2rem; max-width:1450px; }
section[data-testid="stSidebar"] {
    background: #171827 !important;
    border-right: 1px solid #303149;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] a {
    color: #ffffff !important;
}
.glass {
 background:rgba(255,255,255,.62); border:1px solid rgba(255,255,255,.9);
 border-radius:22px; padding:22px; box-shadow:0 10px 30px rgba(67,91,160,.10);
 backdrop-filter:blur(14px); margin-bottom:14px;
}
.hero {
 background:linear-gradient(115deg,rgba(255,255,255,.92),rgba(194,220,255,.77),rgba(222,196,255,.78));
 border:1px solid rgba(255,255,255,.95); border-radius:28px; padding:30px;
 box-shadow:0 14px 35px rgba(78,90,190,.15); margin-bottom:18px;
}
.eyebrow { letter-spacing:3px; font-size:12px; font-weight:800; color:#5268a7; }
.hero h1 { font-size:clamp(36px,5vw,62px); line-height:1.03; margin:8px 0; color:#101b59; }
.hero h1 span { color:#7045f5; }
.hero p { font-size:17px; color:#40588f; max-width:690px; }
.pill { display:inline-block; background:linear-gradient(90deg,#e8ddff,#d8efff); border:1px solid #fff; border-radius:999px; padding:7px 12px; margin:3px; color:#4e40b6; font-size:12px; font-weight:700; }
.metric { background:rgba(255,255,255,.7); border:1px solid white; border-radius:18px; padding:17px; box-shadow:0 6px 18px rgba(67,91,160,.08); min-height:110px; }
.metric .num { font-size:26px; font-weight:800; color:#6845ef; }
.metric .label { color:#53699e; font-size:13px; }
.section-title { font-size:23px; font-weight:800; color:#142260; margin-bottom:4px; }
.small-muted { color:#53699e; font-size:13px; }
div.stButton > button { border-radius:13px; border:1px solid rgba(255,255,255,.9); background:linear-gradient(100deg,#8050ff,#38a8ff); color:white; font-weight:700; min-height:42px; }
div.stButton > button:hover { border-color:#8061ff; color:white; box-shadow:0 5px 15px #8da7ff66; }
.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea { border-radius:13px!important; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.68); border:1px solid white; padding:14px; border-radius:16px; }
hr { border-color:rgba(110,130,200,.18); }
.stTextInput input,
.stTextArea textarea,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: #262630 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

MOLECULES = [
    {"name":"Aspirin","category":"Analgesic","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","note":"Common analgesic and anti-inflammatory agent."},
    {"name":"Caffeine","category":"Stimulant","smiles":"CN1C(=O)N(C)c2ncn(C)c2C1=O","note":"Naturally occurring stimulant found in coffee and tea."},
    {"name":"Paracetamol","category":"Analgesic","smiles":"CC(=O)NC1=CC=C(O)C=C1","note":"Analgesic and antipyretic compound."},
    {"name":"Ibuprofen","category":"NSAID","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","note":"Non-steroidal anti-inflammatory compound."},
    {"name":"Lidocaine","category":"Local anaesthetic","smiles":"CCN(CC)CC(=O)NC1=C(C)C=CC=C1C","note":"Local anaesthetic used in clinical settings."},
    {"name":"Amoxicillin","category":"Antibiotic","smiles":"CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](C3=CC=C(O)C=C3)N)C(=O)N2[C@H]1C(=O)O","note":"Beta-lactam antibiotic; structure shown for education."},
    {"name":"Metformin","category":"Antidiabetic","smiles":"CN(C)C(=N)NC(=N)N","note":"Biguanide compound used in diabetes treatment."},
    {"name":"Nicotinamide","category":"Vitamin-related","smiles":"NC(=O)C1=CC=CN=C1","note":"Amide form related to vitamin B3."},
    {"name":"Warfarin","category":"Anticoagulant","smiles":"CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O","note":"Vitamin K antagonist; educational structure example."},
    {"name":"Fluoxetine","category":"Antidepressant","smiles":"CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1","note":"Selective serotonin reuptake inhibitor."},
    {"name":"Diphenhydramine","category":"Antihistamine","smiles":"CN(C)CCOC(c1ccccc1)c1ccccc1","note":"First-generation antihistamine."},
    {"name":"Atorvastatin","category":"Statin","smiles":"CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)c(-c2ccccc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O","note":"Lipid-lowering medicine; complex structure."},
    {"name":"Nicotine","category":"Alkaloid","smiles":"CN1CCC[C@H]1c1cccnc1","note":"Naturally occurring alkaloid."},
    {"name":"Menthol","category":"Terpenoid","smiles":"CC(C)C1CCC(C)CC1O","note":"A cyclic monoterpenoid alcohol."},
    {"name":"Vanillin","category":"Aromatic compound","smiles":"COc1cc(C=O)ccc1O","note":"Aromatic compound responsible for vanilla aroma."},
]
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def descriptors(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 2),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 3),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
    }

def draw_svg(mol, size=300):
    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return drawer.GetDrawingText()

with st.sidebar:
    st.markdown("## 🧬 MoleculeLab")
    st.caption("Molecular Intelligence Studio")
    page = st.radio("NAVIGATION", ["Home", "Molecule Explorer", "Structure Visualizer", "Molecular Descriptors", "Molecule Comparison", "Data & Analytics", "Learning Centre", "Properties Report", "About"])
    st.markdown("---")
    st.markdown("<div class='glass'><i>Explore molecules.<br>Understand chemistry.<br>Build your future ♡</i></div>", unsafe_allow_html=True)
    st.caption("Educational tool • Not for clinical decisions")

st.markdown("<div class='small-muted'>🔎 Search molecules by name or SMILES using the Molecule Explorer.</div>", unsafe_allow_html=True)

if page == "Home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">WELCOME TO</div>
      <h1>Molecule<span>Lab</span></h1>
      <h3 style="color:#53699e;font-weight:500;margin-top:0">Molecular Intelligence Studio</h3>
      <p>An interactive platform for molecular visualization, structure analysis and drug-property exploration.</p>
      <span class="pill">MOLECULAR EXPLORATION</span><span class="pill">DESCRIPTOR ANALYSIS</span><span class="pill">LEARN BY DOING</span>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,num,label in zip([c1,c2,c3,c4],["15+","8","Interactive","Learn"],["Curated example molecules","Calculated descriptors","Charts & comparisons","Chemoinformatics quizzes"]):
        with col: st.markdown(f"<div class='metric'><div class='num'>{num}</div><div class='label'>{label}</div></div>",unsafe_allow_html=True)
    st.write("")
    left,right=st.columns([1.15,1])
    with left:
        st.markdown("<div class='glass'><div class='section-title'>✨ Quick Molecule Search</div><p class='small-muted'>Choose a molecule to explore its structure and calculated properties.</p></div>",unsafe_allow_html=True)
        selected=st.selectbox("Popular molecules", [m["name"] for m in MOLECULES])
        if st.button("Explore selected molecule →", use_container_width=True):
            st.session_state["selected_molecule"]=selected
            st.session_state["page_from_home"]=True
            st.info("Open Molecule Explorer in the sidebar to view the full analysis.")
    with right:
        m=get_mol(MOLECULES[0]["smiles"])
        st.markdown("<div class='glass'><div class='section-title'>✦ Featured molecule</div><h3>Aspirin</h3><p class='small-muted'>A familiar example for learning molecular descriptors.</p></div>",unsafe_allow_html=True)
        st.image("data:image/svg+xml;utf8," + draw_svg(m), use_container_width=True)
        st.dataframe(pd.DataFrame(list(descriptors(m).items()),columns=["Property","Value"]),hide_index=True,use_container_width=True)
    st.markdown("<div class='section-title'>Explore your lab</div>",unsafe_allow_html=True)
    cards=st.columns(4)
    for col,title,desc in zip(cards,["Structure Visualizer","Molecular Descriptors","Data & Analytics","Learning Centre"],["Generate 2D structures","Calculate key properties","Explore descriptor patterns","Test your understanding"]):
        with col: st.markdown(f"<div class='metric'><div class='num'>✧</div><b>{title}</b><p class='label'>{desc}</p></div>",unsafe_allow_html=True)

elif page == "Molecule Explorer":
    st.markdown("<div class='hero'><div class='eyebrow'>DISCOVER THE CHEMICAL SPACE</div><h1>Molecule <span>Explorer</span></h1><p>Search a curated set of example compounds and inspect their structures.</p></div>",unsafe_allow_html=True)
    query=st.text_input("Search by molecule name, category, or SMILES")
    category=st.selectbox("Filter by category",["All"]+sorted(set(x["category"] for x in MOLECULES)))
    filtered=[x for x in MOLECULES if (query.lower() in (x["name"]+" "+x["category"]+" "+x["smiles"]).lower()) and (category=="All" or x["category"]==category)]
    if not filtered: st.warning("No matching molecules found.")
    for item in filtered:
        with st.expander(f"🧬 {item['name']}  ·  {item['category']}"):
            mol=get_mol(item["smiles"])
            if mol:
                a,b=st.columns([1,1.4])
                with a: st.components.v1.html(draw_svg(mol, 350), height=300, scrolling=False)
                with b:
                    st.write(item["note"])
                    st.code(item["smiles"],language=None)
                    st.dataframe(pd.DataFrame(list(descriptors(mol).items()),columns=["Property","Value"]),hide_index=True,use_container_width=True)
            else: st.error("This example's SMILES could not be parsed; verify the source structure.")

elif page == "Structure Visualizer":
    st.markdown("<div class='hero'><div class='eyebrow'>STRUCTURE WORKSPACE</div><h1>Structure <span>Visualizer</span></h1><p>Enter a SMILES string to render a 2D molecular structure.</p></div>",unsafe_allow_html=True)
    default=MOLECULES[0]["smiles"]
    smiles=st.text_area("SMILES input",value=default,height=90)
    if st.button("Visualize structure"):
        mol=get_mol(smiles.strip())
        if mol:
            st.success("SMILES parsed successfully.")
            st.components.v1.html(draw_svg(mol, 350), height=300, scrolling=False)
            st.code(Chem.MolToSmiles(mol))
        else: st.error("Invalid SMILES. Check the notation and try again.")

elif page == "Molecular Descriptors":
    st.markdown("<div class='hero'><div class='eyebrow'>MOLECULAR PROPERTY WORKBENCH</div><h1>Molecular <span>Descriptors</span></h1><p>Calculate common 2D descriptors from a SMILES string.</p></div>",unsafe_allow_html=True)
    names=[x["name"] for x in MOLECULES]
    name=st.selectbox("Choose an example",names)
    custom=st.checkbox("Use my own SMILES instead")
    smiles=st.text_input("SMILES",MOLECULES[names.index(name)]["smiles"] if not custom else "")
    if smiles:
        mol=get_mol(smiles)
        if mol:
            st.dataframe(pd.DataFrame(list(descriptors(mol).items()),columns=["Descriptor","Calculated value"]),hide_index=True,use_container_width=True)
        else: st.error("Could not parse this SMILES.")

elif page == "Molecule Comparison":
    st.markdown("<div class='hero'><div class='eyebrow'>SIDE-BY-SIDE ANALYSIS</div><h1>Molecule <span>Comparison</span></h1><p>Compare calculated descriptors for two or more example molecules.</p></div>",unsafe_allow_html=True)
    names=[x["name"] for x in MOLECULES]
    selected=st.multiselect("Select molecules",names,default=["Aspirin","Caffeine"],max_selections=6)
    rows=[]
    for name in selected:
        item=next(x for x in MOLECULES if x["name"]==name); mol=get_mol(item["smiles"])
        if mol: rows.append({"Molecule":name,**descriptors(mol)})
    if rows:
        df=pd.DataFrame(rows); st.dataframe(df,hide_index=True,use_container_width=True)
        metric=st.selectbox("Chart descriptor",["Molecular weight (g/mol)","cLogP","TPSA (Å²)","H-bond donors","H-bond acceptors","Rotatable bonds"])
        st.plotly_chart(px.bar(df,x="Molecule",y=metric,color="Molecule",title=f"Comparison: {metric}"),use_container_width=True)

elif page == "Data & Analytics":
    st.markdown("<div class='hero'><div class='eyebrow'>EXPLORE THE DATA</div><h1>Data & <span>Analytics</span></h1><p>Inspect descriptor patterns across example molecules.</p></div>",unsafe_allow_html=True)
    rows=[]
    for item in MOLECULES:
        mol=get_mol(item["smiles"])
        if mol: rows.append({"Molecule":item["name"],"Category":item["category"],**descriptors(mol)})
    df=pd.DataFrame(rows)
    metric=st.selectbox("Choose descriptor",["Molecular weight (g/mol)","cLogP","TPSA (Å²)","H-bond donors","H-bond acceptors","Rotatable bonds"])
    st.plotly_chart(px.scatter(df,x="Molecular weight (g/mol)",y=metric,color="Category",hover_name="Molecule",title=f"{metric} vs molecular weight"),use_container_width=True)
    st.dataframe(df,hide_index=True,use_container_width=True)

elif page == "Learning Centre":
    st.markdown("<div class='hero'><div class='eyebrow'>LEARN BY EXPLORING</div><h1>Learning <span>Centre</span></h1><p>Review core concepts and check your understanding.</p></div>",unsafe_allow_html=True)
    with st.expander("What is SMILES?"):
        st.write("SMILES (Simplified Molecular Input Line Entry System) is a text notation used to represent molecular structures.")
    with st.expander("What is molecular weight?"):
        st.write("Molecular weight is the sum of the average atomic weights of the atoms represented in a molecule.")
    with st.expander("What does cLogP estimate?"):
        st.write("cLogP estimates a compound's partitioning between an octanol-like phase and water; it is a model descriptor, not a direct measurement.")
    with st.expander("What is TPSA?"):
        st.write("Topological polar surface area estimates the contribution of polar atoms and functional groups to molecular surface polarity.")
    questions=[
      ("SMILES is primarily used to…",["Represent molecular structures as text","Measure blood pressure","Predict a patient's diagnosis"],0),
      ("TPSA is a descriptor related to…",["Polar surface area","Colour","Boiling time only"],0),
      ("cLogP is an estimate of…",["Lipophilicity/partitioning","Molecular age","Clinical efficacy"],0),
      ("A descriptor value alone can prove a drug is safe.",["True","False"],1)
    ]
    with st.form("quiz"):
        answers=[]
        for i,(q,opts,correct) in enumerate(questions):
            answers.append(st.radio(q,opts,key=f"q{i}"))
        submitted=st.form_submit_button("Check answers")
    if submitted:
        score=sum(a==q[1][q[2]] for a,q in zip(answers,questions))
        st.success(f"You scored {score} / {len(questions)}.")

elif page == "Properties Report":
    st.markdown("<div class='hero'><div class='eyebrow'>EXPORT YOUR ANALYSIS</div><h1>Properties <span>Report</span></h1><p>Download calculated descriptors for the example molecule set.</p></div>",unsafe_allow_html=True)
    rows=[]
    for item in MOLECULES:
        mol=get_mol(item["smiles"])
        if mol: rows.append({"Molecule":item["name"],"Category":item["category"],"SMILES":Chem.MolToSmiles(mol),**descriptors(mol)})
    df=pd.DataFrame(rows)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.download_button("⬇ Download CSV report",df.to_csv(index=False).encode("utf-8"),"moleculelab_properties.csv","text/csv",use_container_width=True)

else:
    st.markdown("<div class='hero'><div class='eyebrow'>ABOUT THE PROJECT</div><h1>About <span>MoleculeLab</span></h1><p>An educational chemoinformatics workspace for learning how molecular structures connect to computed descriptors.</p></div>",unsafe_allow_html=True)
    st.markdown("<div class='glass'><div class='section-title'>Project purpose</div><p>This student project demonstrates molecular representation, 2D structure drawing, basic descriptor calculation, data comparison, and introductory cheminformatics concepts.</p><div class='section-title'>Important limitations</div><p>Computed descriptors are educational outputs, not clinical evidence. They do not establish efficacy, safety, dosage, or suitability for any person. Check molecular structures and calculations against reliable references before academic submission.</p></div>",unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><span class='small-muted'>MoleculeLab · Molecular Intelligence Studio · Built for learning, not clinical decisions</span></center>",unsafe_allow_html=True)
