import streamlit as st
import pandas as pd
# ---------- 页面基本设置 ----------
st.set_page_config(page_title="Produktbaukästen & Produktportfolios", layout="wide")
st.title("Modellierung der Produktportfolios während des Übergangs zwischen zwei Produktbaukästen")
st.markdown("""
In diesem Tool unterscheiden wir zwei Produktbaukästen – 
zum Beispiel den Modularen Querbaukasten (MQB) und den Modularen E-Antriebs-Baukasten (MEB) – 
und definieren die für jeden Baukasten zugeordneten Produkte 
(z. B. enthält der alte Baukasten MQB Produkte wie Golf, Passat, Tiguan usw.)
""")

# ---------- 帮助函数：把多行文本解析成列表 ----------
def parse_list(text: str):
    """把多行文本拆成列表，自动去掉空行和首尾空格。"""
    return [line.strip() for line in text.splitlines() if line.strip()]

# ---------- 两个 Baukasten 的输入区域 ----------
st.subheader("1. Definition der beiden Baukästen und ihrer Produkte")
col1, col2 = st.columns(2)
with col1:
    # Baukasten 1：默认 MQB
    bk1_name = st.text_input("Baukasten(alt)", value="MQB")
    bk1_products_text = st.text_area(
        "Produkte im alten Baukasten",
        value="Golf\nPassat\nTiguan",
        height=150,
    )
with col2:
    # Baukasten 2：默认 MEB
    bk2_name = st.text_input("Baukasten(neu)", value="MEB")
    bk2_products_text = st.text_area(
        "Produkte im neuen Baukasten",
        value="ID.3\nID.4\nID.5",
        height=150,
    )
# ---------- 解析输入 ----------将输入的文本转换成列表格式
bk1_products = parse_list(bk1_products_text)
bk2_products = parse_list(bk2_products_text)

# 可选：把结果存到 session_state，后面步骤会用到
st.session_state["baukasten_names"] = [bk1_name, bk2_name]
st.session_state["bk1_products"] = bk1_products
st.session_state["bk2_products"] = bk2_products
# ---------- 汇总展示：Baukasten × Produkt ----------

st.markdown("Der alte und der neue Baukasten und deren Produkte (Vorschau)")


if not bk1_products and not bk2_products:
    st.info("Bitte geben Sie in mindestens einen der Baukästen mindestens ein Produkt ein.")
else:
    # 计算两边产品的最大行数
    max_len = max(len(bk1_products), len(bk2_products))
 # 用空字符串把短的一列补齐到一样长
    col_bk1 = bk1_products + [""] * (max_len - len(bk1_products))
    col_bk2 = bk2_products + [""] * (max_len - len(bk2_products))
 # 用 Baukasten 名称作为列名，产品列表作为列数据
    df_portfolio = pd.DataFrame({
        bk1_name: col_bk1,
        bk2_name: col_bk2,
    })
    df_portfolio.index = df_portfolio.index + 1
    st.dataframe(df_portfolio, use_container_width=True)

    st.markdown("""   
    Die obige Tabelle zeigt das aktuelle Produktportfolio sowie den Baukasten, dem jedes Produkt zugeordnet ist.
    In den folgenden Schritten erweitern wir diese Grundlage um:
    - die Produkt × Modul-(Hauptkomponenten)-Matrix,
    - die Plattform/Baukasten × Modul-Matrix,
    - sowie die KPI-Berechnung zur Bewertung der Übergangsphase.
    """)

# ========== 新增：定义两个 Baukasten 中的模块（Hauptkomponenten） ==========
st.subheader("2. Definition der Module in den beiden Baukästen")

st.markdown(f"""
Unten steht eine editierbare Tabelle zur Definition von:   
- **{bk1_name}-spezifischen Modulen**（nur im {bk1_name} verwendet）
- **{bk2_name}-spezifischen Modulen**（nur im {bk2_name} verwendet）
- **gemeinsamen Modulen**（in beiden Baukästen verwendet）
""")

# =============== 初始化：第一次进入页面时创建一个示例表格 =================
if "module_df" not in st.session_state:
    # 给一个示例，足够组合出 Golf / Passat / Tiguan / ID.3 / ID.4 / ID.5
    data_init = {
        "bk1-spezifische Module": [
            "Verbrennungsmotor-Modul 1.5 TSI",
            "Verbrennungsmotor-Modul 2.0 TSI",
            "Verbrennungsmotor-Modul 2.0 TDI",
            "Getriebemodul DQ200",
            "Getriebemodul DQ380",
            "MQB-Karosseriestrukturmodul (Limousine)",
            "MQB-Karosseriestrukturmodul (SUV)",
        ],
        "bk2-spezifische Module": [
            "Hochvoltbatteriemodul 58 kWh",
            "Hochvoltbatteriemodul 77 kWh",
            "E-Motor-Modul Heckantrieb 150 kW",
            "E-Motor-Modul Dualmotor-Allrad 195 kW",
            "MEB-Karosseriestrukturmodul (Limousine)",
            "MEB-Karosseriestrukturmodul (SUV)",
            "",
        ],
        "Gemeinsame Module": [
            "Bremssystem-Modul",
            "Lichtsystem-Modul",
            "Infotainment-Systemmodul MIB1",
            "Infotainment-Systemmodul MIB2",
            "",
            "",
            "",
        ],
    }
    df_init = pd.DataFrame(data_init)
    # 行号从 1 开始（只是好看，不参与计算）
    df_init.index = range(1, len(df_init) + 1)
    st.session_state.module_df = df_init

# 当前 DataFrame（会随着用户编辑而更新）
current_df = st.session_state.module_df

# =============== 可编辑表格 ===============
edited_df = st.data_editor(
    current_df,
    num_rows="dynamic",          # 允许用户增删行
    use_container_width=True,
    key="module_editor",
)

# =============== 统计按钮 ===============
if st.button(
    "Modulbearbeitung abschließen und Statistik erstellen",
    key="button_1"
):
    # 把用户编辑后的表格写回 session_state
    st.session_state.module_df = edited_df
    # 去掉全为空行（防止统计时算进去）
    df_clean = edited_df.copy()
    df_clean = df_clean.dropna(how="all")  # 全 NaN 的行
    # 把 NaN 转成空字符串，方便后面 strip 判断
    df_clean = df_clean.fillna("")
    # 列名：通用写法，不再使用 meb/mqb 作为变量名
    col_bk1_specific = "bk1-spezifische Module"
    col_bk2_specific = "bk2-spezifische Module"
    col_shared       = "Gemeinsame Module"

    def count_non_empty(series: pd.Series) -> int:
        # 统计一列中非空单元格数量
        return series.astype(str).str.strip().ne("").sum()

    # 各列非空单元格数量（平台专用 + 共用）
    n_bk1_specific = count_non_empty(df_clean[col_bk1_specific])
    n_bk2_specific = count_non_empty(df_clean[col_bk2_specific])
    n_shared       = count_non_empty(df_clean[col_shared])

    # Baukasten 1 / 2 的总模块数 = 自己专用 + 共用
    n_bk1_total = n_bk1_specific + n_shared
    n_bk2_total = n_bk2_specific + n_shared

    # 统计总的“不同模块个数”（去重）
    all_values = []
    for col in [col_bk1_specific, col_bk2_specific, col_shared]:
        all_values.extend(df_clean[col].astype(str).str.strip().tolist())
    all_values = [v for v in all_values if v != ""]
    n_unique_total = len(set(all_values))

    st.subheader("Statistik der Modulanzahl")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Gesamtzahl Module in {bk1_name}", int(n_bk1_total))
    c2.metric(f"Gesamtzahl Module in {bk2_name}", int(n_bk2_total))
    c3.metric("Anzahl gemeinsamer Module", int(n_shared))
    c4.metric("Anzahl verschiedener Module (einzigartig)", int(n_unique_total))

    st.markdown("""
    Diese Kennzahlen können anschließend zur Bewertung der Komplexität und Kosten in der Übergangsphase zwischen den beiden Baukästen verwendet werden.
""")

st.markdown("---")

st.subheader("3. Zuordnung von Produkten zu Modulen (Produkt–Modul-Matrix)")

st.markdown("""
In diesem Schritt wird festgelegt, **welche Module von welchem Produkt verwendet werden**.  
""")

# ---------- 初始化 session_state（只在第一次使用时） ----------
if "pxb_started" not in st.session_state:
    st.session_state["pxb_started"] = False          # 是否已经进入产品-模块步骤
if "pxb_current_idx" not in st.session_state:
    st.session_state["pxb_current_idx"] = 0          # 当前正在编辑的产品索引
if "pxb_matrix" not in st.session_state:
    st.session_state["pxb_matrix"] = None            # 保存最终产品×模块矩阵（bool）
if "pxb_last_saved_product" not in st.session_state:
    st.session_state["pxb_last_saved_product"] = None

# ---------- “下一步”按钮：启动产品-模块步骤 ----------
if st.button("Produkt–Modul-Zuordnung starten", key="btn_start_pxb"):
    st.session_state["pxb_started"] = True

# 只有在已启动的情况下才显示后续界面
if st.session_state["pxb_started"]:
    # ① 如果上一个产品刚保存完，这里提示一次
    last_saved = st.session_state.get("pxb_last_saved_product")
    if last_saved:
        st.success(f"Die Modulzuordnung für **{last_saved}** wurde gespeichert.")
        st.session_state["pxb_last_saved_product"] = None
       # ========== 1. 从模块表中提取三个模块列表（bk1专用 / bk2专用 / 共用） ==========
    module_df = st.session_state.module_df.copy()
    module_df = module_df.dropna(how="all").fillna("")

    col_bk1_specific = "bk1-spezifische Module"
    col_bk2_specific = "bk2-spezifische Module"
    col_shared       = "Gemeinsame Module"

    # 将每一列转换为“非空字符串列表”
    bk1_modules = [str(x).strip() for x in module_df[col_bk1_specific] if str(x).strip() != ""]
    bk2_modules = [str(x).strip() for x in module_df[col_bk2_specific] if str(x).strip() != ""]
    shared_modules = [str(x).strip() for x in module_df[col_shared]       if str(x).strip() != ""]

    # 全部模块（用于最终总览矩阵的列），去重保持顺序
    all_modules = list(dict.fromkeys(bk1_modules + bk2_modules + shared_modules))
    # ========== 2. 构建“产品列表 + 所属Baukasten” ==========
    bk1_products = st.session_state.get("bk1_products", [])
    bk2_products = st.session_state.get("bk2_products", [])
    # 列表元素形式：(Baukasten-Name, Produktname)
    product_pairs = [(bk1_name, p) for p in bk1_products] + [(bk2_name, p) for p in bk2_products]

    if not product_pairs:
        st.warning("Es wurden noch keine Produkte definiert. Bitte zuerst Produkte für die Baukästen eingeben.")
    else:
        # ========== 3. 初始化总览矩阵（只在第一次时） ==========
        if st.session_state["pxb_matrix"] is None:
            index_labels = [p for (_, p) in product_pairs]  # 仅用 Produktnamen 作为行索引
            pxb_df = pd.DataFrame(False, index=index_labels, columns=all_modules)
            st.session_state["pxb_matrix"] = pxb_df

        # 当前正在处理的产品索引
        current_idx = st.session_state["pxb_current_idx"]
        # 仍有未处理的产品 → 进入逐个编辑界面

        if current_idx < len(product_pairs):
            current_bk_name, current_product = product_pairs[current_idx]
            st.markdown(
                f"### Produkt {current_idx + 1} von {len(product_pairs)}: "
                f"**{current_product}** (Baukasten: {current_bk_name})"
            )
            # 根据产品所在的 Baukasten 选择可用模块
            if current_bk_name == bk1_name:
                applicable_modules = bk1_modules + shared_modules
            else:
                applicable_modules = bk2_modules + shared_modules
            # 从总览矩阵中取出该产品当前行（布尔值）
            pxb_matrix = st.session_state["pxb_matrix"]
            row_series = pxb_matrix.loc[current_product]
            # 为当前产品构建一个可编辑的小表：左=模块，右=是否使用（Bool）
            df_product = pd.DataFrame({
                "Modul": applicable_modules,
                "Wird vom Produkt verwendet?": [bool(row_series[m]) for m in applicable_modules],
            })
            df_product.index = range(1, len(df_product) + 1)

            edited_product_df = st.data_editor(
                df_product,
                use_container_width=True,
                key=f"editor_product_{current_idx}",
            )
            # “结束当前产品”按钮
            if st.button("Dieses Produkt abschließen", key=f"btn_finish_product_{current_idx}"):
                # 根据用户勾选结果，更新总览矩阵该产品对应的行
                for _, r in edited_product_df.iterrows():
                    mod_name = r["Modul"]
                    used_flag = bool(r["Wird vom Produkt verwendet?"])
                    pxb_matrix.at[current_product, mod_name] = used_flag

                st.session_state["pxb_matrix"] = pxb_matrix
                st.session_state["pxb_current_idx"] = current_idx + 1
                st.session_state["pxb_last_saved_product"] = current_product
                st.rerun()

        # 所有产品都处理完 → 显示总览矩阵
        else:
            st.success("Alle Produkte wurden bearbeitet. Unten steht die Übersicht der Produkt–Modul-Matrix (MDM).")

            final_matrix = st.session_state["pxb_matrix"].copy()

            # 为了便于在论文中使用，这里再生成一个 ✔ / 0 视图
            view_matrix = final_matrix.replace({True: "✔", False: ""})
            view_matrix.index.name = "Produkt"
            st.dataframe(view_matrix, use_container_width=True)
#             st.markdown("""
# Diese Matrix stellt die **MDM-Sicht (Produkt × Modul)** dar:
# - Zeilen: Produkte aus beiden Baukästen
# - Spalten: alle relevanten Module
# - „✔“ bedeutet, dass das jeweilige Modul vom Produkt verwendet wird.
#
# Auf Basis dieser Matrix können im nächsten Schritt Kennzahlen zur Komplexität und Wirtschaftlichkeit
# für verschiedene Übergangsszenarien (z. B. unterschiedliche Zwischen-Baukasten-Varianten) berechnet werden.
# """)

# ============================================
# 4. Definition des Zwischen-Baukastens (Zwischen 1 & Zwischen 2)
# ============================================

if "zwischen_started" not in st.session_state:
    st.session_state["zwischen_started"] = False

st.markdown("---")
st.subheader("4. Definition des Zwischen-Baukastens (Zwischen 1 & Zwischen 2)")
st.markdown("""
In diesem Schritt werden **zwei verschiedene Zwischen-Baukasten-Szenarien** definiert:

- **Zwischen 1**: erste Auswahl von Produkten  
- **Zwischen 2**: zweite Auswahl von Produkten  

Diese Szenarien können später hinsichtlich ihrer KPI verglichen werden.
""")

# ---------- 前置条件检查：必须完成前面的 Schritte ----------
bk1_products = st.session_state.get("bk1_products", [])
bk2_products = st.session_state.get("bk2_products", [])
has_products = bool(bk1_products or bk2_products)

has_modules = "module_df" in st.session_state
pxb_matrix = st.session_state.get("pxb_matrix", None)
has_pxb = pxb_matrix is not None

if not (has_products and has_modules and has_pxb):
    st.info("Bitte führen Sie zunächst die vorherigen Schritte (1–3) vollständig durch.")
else:
    # 先有一个按钮控制是否展开 Zwischen 步骤
    if not st.session_state["zwischen_started"]:
        if st.button("Zwischen-Baukasten-Szenarien definieren", key="btn_start_zwischen"):
            st.session_state["zwischen_started"] = True

    if st.session_state["zwischen_started"]:

        # ---------- 1. 产品列表（用于两个 Zwischen 方案共用） ----------
        product_rows = []
        for p in bk1_products:
            product_rows.append({"Produkt": p, "Baukasten": bk1_name})
        for p in bk2_products:
            product_rows.append({"Produkt": p, "Baukasten": bk2_name})

        if not product_rows:
            st.info("Es wurden noch keine Produkte definiert. Bitte zuerst Produkte für die Baukästen eingeben.")
        else:
            current_products = [row["Produkt"] for row in product_rows]

            # ========== 2. Zwischen 1 的表格 ==========
            st.markdown("### Zwischen 1 – Produktauswahl")

            # 判断是否需要初始化 / 重建模板
            need_init_zw1 = False
            if "zw1_df_template" not in st.session_state:
                need_init_zw1 = True
            else:
                df_old = st.session_state["zw1_df_template"]
                if df_old["Produkt"].tolist() != current_products:
                    need_init_zw1 = True

            if need_init_zw1:
                init_rows_1 = []
                for row in product_rows:
                    init_rows_1.append(
                        {
                            "Produkt": row["Produkt"],
                            "Baukasten": row["Baukasten"],
                            "Im Zwischen 1?": False,
                        }
                    )
                df_zw1_init = pd.DataFrame(init_rows_1)
                df_zw1_init.index = range(1, len(df_zw1_init) + 1)
                st.session_state["zw1_df_template"] = df_zw1_init

            df_zw1_template = st.session_state["zw1_df_template"]

            edited_zw1 = st.data_editor(
                df_zw1_template,
                use_container_width=True,
                num_rows="fixed",
                key="zwischen1_editor",
            )

            # 从编辑结果中提取 Zwischen 1 的产品选择
            selection_zw1 = {}
            for _, r in edited_zw1.iterrows():
                selection_zw1[r["Produkt"]] = bool(r["Im Zwischen 1?"])

            st.session_state["zwischen1_selection"] = selection_zw1
            zwischen1_products = [p for p, flag in selection_zw1.items() if flag]
            st.session_state["zwischen1_products"] = zwischen1_products

            if not zwischen1_products:
                st.warning("Zwischen 1 enthält derzeit noch keine Produkte.")
            else:
                st.success("Produkte in Zwischen 1: " + ", ".join(zwischen1_products))

            # ========== 3. Zwischen 2 的表格 ==========
            st.markdown("### Zwischen 2 – Produktauswahl")

            need_init_zw2 = False
            if "zw2_df_template" not in st.session_state:
                need_init_zw2 = True
            else:
                df_old2 = st.session_state["zw2_df_template"]
                if df_old2["Produkt"].tolist() != current_products:
                    need_init_zw2 = True

            if need_init_zw2:
                init_rows_2 = []
                for row in product_rows:
                    init_rows_2.append(
                        {
                            "Produkt": row["Produkt"],
                            "Baukasten": row["Baukasten"],
                            "Im Zwischen 2?": False,
                        }
                    )
                df_zw2_init = pd.DataFrame(init_rows_2)
                df_zw2_init.index = range(1, len(df_zw2_init) + 1)
                st.session_state["zw2_df_template"] = df_zw2_init

            df_zw2_template = st.session_state["zw2_df_template"]

            edited_zw2 = st.data_editor(
                df_zw2_template,
                use_container_width=True,
                num_rows="fixed",
                key="zwischen2_editor",
            )

            # 从编辑结果中提取 Zwischen 2 的产品选择
            selection_zw2 = {}
            for _, r in edited_zw2.iterrows():
                selection_zw2[r["Produkt"]] = bool(r["Im Zwischen 2?"])

            st.session_state["zwischen2_selection"] = selection_zw2
            zwischen2_products = [p for p, flag in selection_zw2.items() if flag]
            st.session_state["zwischen2_products"] = zwischen2_products

            if not zwischen2_products:
                st.warning("Zwischen 2 enthält derzeit noch keine Produkte.")
            else:
                st.success("Produkte in Zwischen 2: " + ", ".join(zwischen2_products))

            # ========== 4. 基于 Produkt–Modul-Matrix 计算模块数量 ==========
            pxb = pxb_matrix.copy()
            existing_products = [p for p in pxb.index if p in (bk1_products + bk2_products)]
            pxb = pxb.loc[existing_products]

            def count_modules_for_products(product_list):
                """根据给定的产品列表，计算使用到的不同模块数量。"""
                if not product_list:
                    return 0
                valid = [p for p in product_list if p in pxb.index]
                if not valid:
                    return 0
                used = pxb.loc[valid].any(axis=0)  # 某列有 True 就表示该模块被至少一个产品使用
                return int(used.sum())

            # Alter / Neuer Baukasten
            n_bk1_modules = count_modules_for_products(bk1_products)
            n_bk2_modules = count_modules_for_products(bk2_products)

            # Zwischen 1 / Zwischen 2
            n_zw1_modules = count_modules_for_products(zwischen1_products)
            n_zw2_modules = count_modules_for_products(zwischen2_products)

            # ========== 5. 可视化：比较四个“Baukasten”的模块数量 ==========
            st.markdown("### Vergleich der Modulanzahl (alter BK, Zwischen 1, Zwischen 2, neuer BK)")

            df_bar = pd.DataFrame(
                {
                    "Anzahl unterschiedlicher Module": [
                        n_bk1_modules,
                        n_zw1_modules,
                        n_zw2_modules,
                        n_bk2_modules,
                    ]
                },
                index=[bk1_name, "Zwischen 1", "Zwischen 2", bk2_name],
            )

            st.bar_chart(df_bar)

# ============================================
# 5. KPI-Vergleich der Zwischen-Szenarien
# ============================================

st.markdown("---")
st.subheader("5. KPI-Vergleich der Zwischen-Szenarien")
st.markdown("### 5.1 Ingenieurtechnische KPI")

st.markdown("""
In diesem Abschnitt werden die beiden Zwischen-Baukasten-Szenarien **Zwischen 1** und **Zwischen 2**
hinsichtlich ausgewählter **ingenieurtechnischer Kennzahlen (KPI)** verglichen:

1. **KPI 1 – Modulanzahl**: Wie viele unterschiedliche Module werden im Szenario benötigt?  
2. **KPI 2 – Wiederverwendungsrate**: Welcher Anteil der Module wird aus bestehenden Baukästen wiederverwendet?  
3. **KPI 3 – Ähnlichkeit zum neuen Baukasten**: Wie stark ähnelt das Szenario dem neuen Baukasten hinsichtlich der verwendeten Module?
""")

# ---------- 前置条件检查 ----------
zwischen1_products = st.session_state.get("zwischen1_products", [])
zwischen2_products = st.session_state.get("zwischen2_products", [])
pxb_matrix = st.session_state.get("pxb_matrix", None)
has_module_df = "module_df" in st.session_state

if not (zwischen1_products or zwischen2_products):
    st.info("Bitte definieren Sie zuerst die Zwischen-Szenarien (Zwischen 1 und/oder Zwischen 2).")
elif pxb_matrix is None or not has_module_df:
    st.info("Für die KPI-Berechnung werden die Moduldefinition (Schritt 3) und die Produkt–Modul-Matrix (Schritt 4) benötigt.")
else:
    # ========== 1. 从模块表中提取 bk1 / bk2 / 共用模块 ==========
    module_df = st.session_state["module_df"].copy()
    module_df = module_df.dropna(how="all").fillna("")

    col_bk1_specific = "bk1-spezifische Module"
    col_bk2_specific = "bk2-spezifische Module"
    col_shared       = "Gemeinsame Module"

    bk1_modules = [str(x).strip() for x in module_df[col_bk1_specific] if str(x).strip() != ""]
    bk2_modules = [str(x).strip() for x in module_df[col_bk2_specific] if str(x).strip() != ""]
    shared_modules = [str(x).strip() for x in module_df[col_shared]     if str(x).strip() != ""]

    # 共享模块视为两个 Baukästen 都在使用
    modules_bk1_total = set(bk1_modules) | set(shared_modules)
    modules_bk2_total = set(bk2_modules) | set(shared_modules)
    modules_all_base  = modules_bk1_total | modules_bk2_total   # “已有模块池”，用于算复用率

    # ========== 2. 辅助函数：根据产品集合求“该方案用到的模块集合” ==========
    def modules_for_products(products, pxb: pd.DataFrame) -> set:
        """给定一个产品列表，从 Produkt–Modul-Matrix 中求出该方案使用到的所有模块集合。"""
        if not products:
            return set()
        valid = [p for p in products if p in pxb.index]
        if not valid:
            return set()
        used_cols = pxb.loc[valid].any(axis=0)  # 某列只要有一个 True，就表示该模块被用了
        return set(used_cols[used_cols].index)

    pxb = pxb_matrix.copy()
    modules_zw1 = modules_for_products(zwischen1_products, pxb)
    modules_zw2 = modules_for_products(zwischen2_products, pxb)

    # ========== 3. 计算三个工程类 KPI ==========
    def compute_engineering_kpis(modules_zw: set):
        """
        返回 (模块总数, 模块复用率, 平台接近度_BK2)

        - 模块总数: |Module_ZW|
        - 模块复用率: |Module_ZW ∩ (Module_BK1_total ∪ Module_BK2_total)| / |Module_ZW|
        - 平台接近度: |Module_ZW ∩ Module_BK2_total| / |Module_ZW|
        """
        count = len(modules_zw)
        if count == 0:
            return 0, 0.0, 0.0

        reused = len(modules_zw & modules_all_base) / count
        similarity_bk2 = len(modules_zw & modules_bk2_total) / count
        return count, reused, similarity_bk2

    k1_zw1, k2_zw1, k3_zw1 = compute_engineering_kpis(modules_zw1)
    k1_zw2, k2_zw2, k3_zw2 = compute_engineering_kpis(modules_zw2)

    # ========== 4. KPI 1：文字 + 公式 + 柱状图 ==========
    st.markdown("#### KPI 1 – Modulanzahl")

    st.markdown("""
**Definition:** Anzahl der unterschiedlichen Module, die im jeweiligen Zwischen-Szenario verwendet werden.  
Je weniger Module benötigt werden, desto niedriger sind Entwicklungs-, Test-, Logistik- und Lageraufwände.
""")
    st.latex(r"KPI_{1}(ZW) = \left| Module_{ZW} \right|")

    df_kpi1 = pd.DataFrame(
        {"KPI 1 – Modulanzahl": [k1_zw1, k1_zw2]},
        index=["Zwischen 1", "Zwischen 2"],
    )
    st.bar_chart(df_kpi1)

    # ========== 5. KPI 2：文字 + 公式 + 柱状图 ==========
    st.markdown("#### KPI 2 – Wiederverwendungsrate")

    st.markdown(f"""
**Definition:** Anteil der Module im Zwischen-Szenario, die bereits in den bestehenden Baukästen  
({bk1_name}, {bk2_name} bzw. gemeinsamen Modulen) verwendet werden.  

Je höher diese Rate, desto besser werden bestehende Module wiederverwendet und desto geringer ist der Bedarf an Neuentwicklungen.
""")
    st.latex(
        r"KPI_{2}(ZW) = \frac{\left| Module_{ZW} \cap "
        r"\left(Module_{BK1}^{gesamt} \cup Module_{BK2}^{gesamt}\right) \right|}"
        r"{\left| Module_{ZW} \right|}"
    )

    df_kpi2 = pd.DataFrame(
        {"KPI 2 – Wiederverwendungsrate": [k2_zw1, k2_zw2]},
        index=["Zwischen 1", "Zwischen 2"],
    )
    st.bar_chart(df_kpi2)

    # ========== 6. KPI 3：文字 + 公式 + 柱状图 ==========
    st.markdown("#### KPI 3 – Ähnlichkeit zum neuen Baukasten")

    st.markdown(f"""
**Definition:** Anteil der im Zwischen-Szenario verwendeten Module, die auch im neuen Baukasten ({bk2_name})  
verwendet werden (inklusive gemeinsamer Module).  

Je höher dieser Wert, desto „näher“ liegt das Szenario am Zielzustand des neuen Baukastens.
""")
    st.latex(
        r"KPI_{3}(ZW) = \frac{\left| Module_{ZW} \cap Module_{BK2}^{gesamt} \right|}"
        r"{\left| Module_{ZW} \right|}"
    )

    df_kpi3 = pd.DataFrame(
        {"KPI 3 – Ähnlichkeit zum neuen Baukasten": [k3_zw1, k3_zw2]},
        index=["Zwischen 1", "Zwischen 2"],
    )
    st.bar_chart(df_kpi3)

    # ========== 7. 小结表格（可选，保留一个总览） ==========
    st.markdown("##### Übersicht (numerische Werte)")

    df_kpi_overview = pd.DataFrame(
        {
            "Zwischen 1": [k1_zw1, k2_zw1, k3_zw1],
            "Zwischen 2": [k1_zw2, k2_zw2, k3_zw2],
        },
        index=[
            "KPI 1: Modulanzahl",
            "KPI 2: Wiederverwendungsrate",
            "KPI 3: Ähnlichkeit zu neuem Baukasten",
        ],
    )
    st.dataframe(df_kpi_overview, use_container_width=True)

    # ============================================
    # 5.1 Normalisierung der ingenieurtechnischen KPI
    # ============================================

    st.markdown("#### Normalisierung der ingenieurtechnischen KPI")

    st.markdown(r"""
    Um die KPI vergleichbar zu machen werden sie auf den Bereich $[0,1]$ abgebildet und in die gleiche Richtung
    (**je größer desto besser**) transformiert.

    **1. Richtungsvereinheitlichung**

    Wir transformieren ihn in eine Nutzenkennzahl

    $$
    KPI_1^{(+)}(ZW_i)
    = \frac{\min\big(KPI_1(ZW_1),\,KPI_1(ZW_2)\big)}{KPI_1(ZW_i)}
    $$

    sodass größere Werte einer geringeren Modulanzahl (also einer besseren Ausprägung) entsprechen.

    **2. Skalierung auf $[0,1]$**

    Für KPI&nbsp;1 skalieren wir diese Nutzenkennzahl so, dass das beste Zwischen-Szenario den Wert 1 erhält:

    $$
    KPI_1^{norm}(ZW_i)
    = \frac{KPI_1^{(+)}(ZW_i)}{\max_j KPI_1^{(+)}(ZW_j)} \in [0,1]
    $$

    Für KPI&nbsp;2 (Wiederverwendungsrate) und KPI&nbsp;3 (Ähnlichkeit zum neuen Baukasten)
    liegen die ursprünglichen Werte bereits im Intervall $[0,1]$ und sind nutzenorientiert, daher setzen wir

    $$
    KPI_2^{norm}(ZW_i) = KPI_2(ZW_i), \qquad
    KPI_3^{norm}(ZW_i) = KPI_3(ZW_i)
    $$

    und interpretieren sie direkt als normalisierte Werte.
    """)

    # -------- 1. KPI 1: von „weniger ist besser“ zu „mehr ist besser“ --------
    k1_values = [k1_zw1, k1_zw2]  # Modulanzahl in Zwischen 1 / 2
    k1_min = min(k1_values)

    def safe_reverse(x, min_val):
        # 防止除 0：理论上模块数不可能为 0，这里只是保险
        return (min_val / x) if x > 0 else 0.0

    # Richtung vereinheitlicht (größer = besser)
    k1_plus = [safe_reverse(v, k1_min) for v in k1_values]

    # Auf [0,1] skaliert: bestes Szenario = 1, anderes < 1
    k1_max_plus = max(k1_plus) if k1_plus else 1.0
    k1_norm = [v / k1_max_plus for v in k1_plus]

    # -------- 2. KPI 2 und KPI 3: liegen bereits in [0,1] --------
    # 直接把原始值当作归一化后的值
    k2_norm = [k2_zw1, k2_zw2]  # Wiederverwendungsrate
    k3_norm = [k3_zw1, k3_zw2]  # Ähnlichkeit zum neuen BK

    # -------- 3. Zusammenstellung in einer Übersichtstabelle --------
    df_kpi_norm = pd.DataFrame(
        {
            "Zwischen 1": [k1_norm[0], k2_norm[0], k3_norm[0]],
            "Zwischen 2": [k1_norm[1], k2_norm[1], k3_norm[1]],
        },
        index=[
            "KPI 1: Modulanzahl (normalisiert)",
            "KPI 2: Wiederverwendungsrate (normalisiert)",
            "KPI 3: Ähnlichkeit zum neuen Baukasten (normalisiert)",
        ],
    )

    st.markdown(r"""
    Die nachfolgende Tabelle zeigt die **normalisierten Ingenieur-KPI**.
    Alle Werte liegen im Bereich $[0,1]$, wobei höhere Werte jeweils eine **bessere Ausprägung** des KPI bedeuten.
    """)
    st.dataframe(df_kpi_norm, use_container_width=True)

    st.markdown("###### Grafischer Vergleich der normalisierten Ingenieur-KPI")
    st.bar_chart(df_kpi_norm.T)

    # -------- 4. Gesamter Ingenieur-KPI (als einfacher Durchschnitt) --------
    # 目前先用三个归一化 KPI 的算术平均，后续如果需要可以引入权重 w1, w2, w3
    eng_score_zw1 = (k1_norm[0] + k2_norm[0] + k3_norm[0]) / 3.0
    eng_score_zw2 = (k1_norm[1] + k2_norm[1] + k3_norm[1]) / 3.0

    df_eng_total = pd.DataFrame(
        {
            "Zwischen 1": [eng_score_zw1],
            "Zwischen 2": [eng_score_zw2],
        },
        index=["Gesamt-KPI (Ingenieur)"],
    )

    st.markdown(r"""
    **Gesamter Ingenieur-KPI**

    Zur weiteren Auswertung fassen wir die drei normalisierten Ingenieur-KPI zu einem
    einzigen **Ingenieur-Gesamtindex** zusammen (hier zunächst als einfacher Durchschnitt):

    $$
    KPI_{\text{Ing}}(ZW_i)
    = \frac{KPI_1^{norm}(ZW_i) + KPI_2^{norm}(ZW_i) + KPI_3^{norm}(ZW_i)}{3}
    $$

    Dieser Gesamtwert liegt ebenfalls im Intervall $[0,1]$ und kann später mit einem
    wirtschaftlichen Gesamt-KPI kombiniert werden.
    """)

    st.dataframe(df_eng_total, use_container_width=True)


