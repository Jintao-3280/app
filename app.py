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
st.subheader("2. Der alte und der neue Baukasten und deren Produkte (Vorschau)")

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
st.subheader("3. Definition der Module in den beiden Baukästen")

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

st.subheader("4. Zuordnung von Produkten zu Modulen (Produkt–Modul-Matrix)")

st.markdown("""
In diesem Schritt wird festgelegt, **welche Module von welchem Produkt verwendet werden**.  
Für Produkte im alten Baukasten werden nur Module aus dem alten Baukasten und gemeinsame Module angezeigt,  
für Produkte im neuen Baukasten nur Module aus dem neuen Baukasten und gemeinsame Module.
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
if st.button("Nächster Schritt: Produkt–Modul-Zuordnung starten", key="btn_start_pxb"):
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

# ============================================================
# 5. Definition des Zwischen-Baukastens und Modulanzahl-Vergleich
# ============================================================

# 初始化“是否进入第 5 步”的状态
if "zwischen_started" not in st.session_state:
    st.session_state["zwischen_started"] = False

st.markdown("---")
st.subheader("5. Definition des Zwischen-Baukastens")

# ---------- 先检查前 1–4 步是否完成 ----------
# 1/2 步：至少有产品
bk1_products = st.session_state.get("bk1_products", [])
bk2_products = st.session_state.get("bk2_products", [])
has_products = bool(bk1_products or bk2_products)

# 3 步：模块表已存在（module_df 在你前面初始化时就写入了）
has_modules = "module_df" in st.session_state

# 4 步：Produkt–Modul-Matrix 已生成
pxb_matrix = st.session_state.get("pxb_matrix", None)
has_pxb = pxb_matrix is not None

if not (has_products and has_modules and has_pxb):
    st.info("""
Bitte führen Sie zunächst die vorne Schritte durch.
""")
else:
    # 只有在前 1–4 步都完成的前提下，才显示“开始第 5 步”的按钮
    if not st.session_state["zwischen_started"]:
        if st.button("Nächster Schritt: Zwischen-Baukasten definieren", key="btn_start_zwischen"):
            st.session_state["zwischen_started"] = True

    # 只有用户点击了按钮，才展开第 5 步的所有内容
    if st.session_state["zwischen_started"]:

        st.markdown("""
        In diesem Schritt wird festgelegt, **welche Produkte im Zwischen-Baukasten enthalten sind**.  
        """)

        # ---------- 1. Alle Produkte (inkl. zugehörigem Baukasten) einsammeln ----------
        product_rows = []
        for p in bk1_products:
            product_rows.append({"Produkt": p, "Baukasten": bk1_name})
        for p in bk2_products:
            product_rows.append({"Produkt": p, "Baukasten": bk2_name})

        if not product_rows:
            st.info("Es wurden noch keine Produkte definiert. Bitte zuerst Produkte für die Baukästen eingeben.")
        else:
            # ---------- 2. Zwischen-Auswahl-DataFrame mit gespeicherten Werten vorbereiten ----------
            # 之前用户勾选的 Zwischen 结果（字典：Produkt -> Bool）
            prev_selection = st.session_state.get("zwischen_selection", {})

            for row in product_rows:
                prod_name = row["Produkt"]
                # 默认 False，如果之前有选择则保持
                row["Im Zwischen-Baukasten?"] = bool(prev_selection.get(prod_name, False))

            df_zwischen = pd.DataFrame(product_rows)
            df_zwischen.index = range(1, len(df_zwischen) + 1)

            # ---------- 3. 可编辑表格：选择 Zwischen 中包含哪些产品 ----------
            st.markdown("Wählen Sie in der folgenden Tabelle aus, welche Produkte im Zwischen-Baukasten enthalten sind:")

            edited_zwischen = st.data_editor(
                df_zwischen,
                use_container_width=True,
                num_rows="fixed",   # 行数固定，不允许增删行
                key="zwischen_editor",
            )

            # 把选择结果写回 session_state，方便后续使用
            selection_dict = {}
            for _, r in edited_zwischen.iterrows():
                selection_dict[r["Produkt"]] = bool(r["Im Zwischen-Baukasten?"])
            st.session_state["zwischen_selection"] = selection_dict

            zwischen_products = [p for p, flag in selection_dict.items() if flag]
            st.session_state["zwischen_products"] = zwischen_products

            if not zwischen_products:
                st.warning("Der Zwischen-Baukasten enthält derzeit noch keine Produkte.")
            else:
                st.success(
                    "Produkte im Zwischen-Baukasten: "
                    + ", ".join(zwischen_products)
                )

            # ---------- 4. Modulanzahl für alten BK, Zwischen-BK und neuen BK berechnen ----------
            # 这里的 pxb_matrix 已经在前面 has_pxb 检查过不为 None
            pxb = pxb_matrix.copy()

            # 只保留矩阵中真正存在的产品行（防止产品列表变化导致的缺失）
            existing_products = [p for p in pxb.index if p in (bk1_products + bk2_products)]
            pxb = pxb.loc[existing_products]

            # — Baukasten 1: 所有属于 bk1 的产品使用到的模块合集 —
            if bk1_products:
                used_bk1 = pxb.loc[[p for p in bk1_products if p in pxb.index]].any(axis=0)
                n_bk1_modules = int(used_bk1.sum())
            else:
                n_bk1_modules = 0

            # — Baukasten 2: 同理 —
            if bk2_products:
                used_bk2 = pxb.loc[[p for p in bk2_products if p in pxb.index]].any(axis=0)
                n_bk2_modules = int(used_bk2.sum())
            else:
                n_bk2_modules = 0

            # — Zwischen-Baukasten: 只看被选中的产品 —
            if zwischen_products:
                valid_zw = [p for p in zwischen_products if p in pxb.index]
                if valid_zw:
                    used_zw = pxb.loc[valid_zw].any(axis=0)
                    n_zw_modules = int(used_zw.sum())
                else:
                    n_zw_modules = 0
            else:
                n_zw_modules = 0

            # ---------- 5. 用柱状图展示三个 Baukästen 的模块数量 ----------
            st.markdown("### Vergleich der Modulanzahl (alter BK, Zwischen-BK, neuer BK)")

            df_bar = pd.DataFrame(
                {
                    "Anzahl unterschiedlicher Module": [
                        n_bk1_modules,
                        n_zw_modules,
                        n_bk2_modules,
                    ]
                },
                index=[bk1_name, "Zwischen", bk2_name],
            )

            st.bar_chart(df_bar)
            st.markdown("### Liniengrafik: Modulanzahl im Vergleich")
            st.line_chart(df_bar)
            st.markdown("""
Die Balkengrafik zeigt die Anzahl unterschiedlicher Module, die in den jeweiligen
Baukästen benötigt werden (auf Basis der Produkt–Modul-Matrix):

- **Alter Baukasten**: Module, die von Produkten im alten Baukasten verwendet werden  
- **Zwischen-Baukasten**: Module, die von den ausgewählten Zwischen-Produkten verwendet werden  
- **Neuer Baukasten**: Module, die von Produkten im neuen Baukasten verwendet werden  

Diese Kennzahlen können im weiteren Verlauf zur Bewertung verschiedener
Zwischen-Baukasten-Szenarien herangezogen werden.
""")


st.markdown("""
Nächste Schritte:

Die Einführung einer Plattform zwischen MQB und MEB, die vorläufig „zwischen“ genannt wird.

Betrachtung verschiedener Kombinationen innerhalb von „zwischen“.

Zum Beispiel:

– zwischen 1 umfasst folgende Produkte: Golf, Passat, Tiguan, ID.3, ID.4.

– zwischen 2 umfasst folgende Produkte: Golf, Passat, ID.3, ID.4.

Bewertung der wirtschaftlichen Effizienz dieser beiden Varianten (unter Berücksichtigung verschiedener relevanter Faktoren wie Komplexität, Marge usw.).
Komplexe Gleichungen

Einführung einer Zeitachse.

""")