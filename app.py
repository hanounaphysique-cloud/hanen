import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ne pas enlever
import streamlit as st
import pandas as pd

# --------- CONSTANTES PHYSIQUES ---------
e_charge = 1.602176634e-19
k_B = 1.380649e-23
m_D = 3.344e-27
T_gas = 300.0
S0_DD = 55.0
E_G_DD = 986.0
barn_to_m2 = 1e-28

# --------- PARAMÈTRES NUMÉRIQUES ---------
NUMERIC_PARAMS = ["V_kV", "I_mA", "P_ubar", "Dch_cm", "flow_SCCM", "Dgrid_cm"]

DEFAULT_RANGES = {
    "V_kV":   (10.0, 80.0, 5.0),
    "I_mA":   (1.0, 50.0, 5.0),
    "P_ubar": (20.0, 30.0, 1.0),
    "Dch_cm": (10.0, 35.0, 5.0),
    "flow_SCCM": (0.0, 10.0, 1.0),
    "Dgrid_cm": (1.0, 10.0, 0.5),
}

PARAM_LABELS = {
    "V_kV": "Tension cathode (kV)",
    "I_mA": "Courant cathode (mA)",
    "P_ubar": "Pression gaz (µbar)",
    "Dch_cm": "Diamètre chambre (cm)",
    "flow_SCCM": "Débit gaz (SCCM)",
    "Dgrid_cm": "Diamètre grille (cm)",
}

# --------- FONCTIONS PHYSIQUES ---------
def sigma_DD_bosch_hale(E_cm_keV: float) -> float:
    if E_cm_keV <= 0.1:
        return 0.0
    sigma_barn = (S0_DD / E_cm_keV) * np.exp(-np.sqrt(E_G_DD / E_cm_keV))
    return float(sigma_barn * barn_to_m2)

def gas_density(cfg: dict) -> float:
    P_ubar = cfg["P_ubar"]
    P_Pa = P_ubar * 0.1
    return float(P_Pa / (k_B * T_gas))

def fusion_rate_DD(cfg: dict) -> float:
    V_kV = cfg["V_kV"]
    I_mA = cfg["I_mA"]
    P_ubar = cfg["P_ubar"]
    Dch_cm = cfg["Dch_cm"]
    gas = cfg["gas"]

    if gas == "H2":
        return 0.0

    I_A = I_mA * 1e-3
    P_Pa = P_ubar * 0.1
    Dch_m = Dch_cm / 100.0
    r_m = Dch_m / 2.0

    n_D = P_Pa / (k_B * T_gas)
    E_lab_keV = max(V_kV, 0.1)
    E_cm_keV = 0.5 * E_lab_keV
    sigma = sigma_DD_bosch_hale(E_cm_keV)

    Y = I_A * n_D * sigma * r_m / (3.0 * e_charge)
    return float(Y)

def xray_power(cfg: dict) -> float:
    V_kV = cfg["V_kV"]
    I_mA = cfg["I_mA"]
    I_A = I_mA * 1e-3
    k = 0.01
    return float(k * V_kV * 1e3 * I_A)

QUANTITIES = {
    "Production neutronique": {
        "fn": fusion_rate_DD,
        "metric_label": "Y max ≈ (neutrons/s)",
        "colorbar_label": "Y ~ neutrons/s (échelle relative)",
        "title_2d": "Production neutronique D–D (~neutrons/s)",
        "title_3d": "Empilement de matrices – production neutronique D–D",
    },
    "Densité du gaz": {
        "fn": gas_density,
        "metric_label": "n_D max ≈ (m⁻³)",
        "colorbar_label": "Densité du gaz (m⁻³)",
        "title_2d": "Densité de deutérium dans la chambre",
        "title_3d": "Empilement de matrices – densité du gaz",
    },
    "Puissance X (approx.)": {
        "fn": xray_power,
        "metric_label": "P_X max ≈ (W, approx.)",
        "colorbar_label": "P_X ~ W (échelle relative)",
        "title_2d": "Puissance X approximative",
        "title_3d": "Empilement de matrices – puissance X (approx.)",
    },
}

# --------- MATRICES ---------
def build_matrix(axis_x, axis_y, ranges, fixed_cfg, quantity_fn):
    xmin, xmax, xstep = ranges[axis_x]
    ymin, ymax, ystep = ranges[axis_y]
    x_vals = np.arange(xmin, xmax + 1e-9, xstep)
    y_vals = np.arange(ymin, ymax + 1e-9, ystep)
    M = np.zeros((len(y_vals), len(x_vals)))
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            cfg = fixed_cfg.copy()
            cfg[axis_x] = x
            cfg[axis_y] = y
            M[i, j] = quantity_fn(cfg)
    return x_vals, y_vals, M

def build_matrix_stack(axis_x, axis_y, axis_z, ranges, fixed_cfg, quantity_fn):
    xmin, xmax, xstep = ranges[axis_x]
    ymin, ymax, ystep = ranges[axis_y]
    zmin, zmax, zstep = ranges[axis_z]
    x_vals = np.arange(xmin, xmax + 1e-9, xstep)
    y_vals = np.arange(ymin, ymax + 1e-9, ystep)
    z_vals = np.arange(zmin, zmax + 1e-9, zstep)
    M = np.zeros((len(z_vals), len(y_vals), len(x_vals)))
    for k, z in enumerate(z_vals):
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                cfg = fixed_cfg.copy()
                cfg[axis_x] = x
                cfg[axis_y] = y
                cfg[axis_z] = z
                M[k, i, j] = quantity_fn(cfg)
    return x_vals, y_vals, z_vals, M

# --------- PAGE MATRICES THÉORIQUES ---------
def page_theorie():
    st.title("Matrices théoriques (Bosch–Hale)")

    st.sidebar.header("Paramètres généraux")
    gas = st.sidebar.selectbox("Gaz", ["D2", "H2"], index=0)
    mat_grid = st.sidebar.selectbox("Matériau grille (info)", ["cuivre", "inox"], index=0)
    mode = st.sidebar.radio("Mode d'affichage", ["Matrice 2D", "Empilement 3D"])

    grandeur_name = st.sidebar.selectbox(
        "Grandeur à afficher",
        list(QUANTITIES.keys()),
        index=0
    )
    qinfo = QUANTITIES[grandeur_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Choix des axes")
    axis_x = st.sidebar.selectbox("Axe X", NUMERIC_PARAMS, index=0, format_func=lambda k: PARAM_LABELS[k])
    axis_y = st.sidebar.selectbox("Axe Y", NUMERIC_PARAMS, index=1, format_func=lambda k: PARAM_LABELS[k])

    axis_z = None
    if mode == "Empilement 3D":
        axis_z = st.sidebar.selectbox("Axe Z (couches)", NUMERIC_PARAMS, index=2, format_func=lambda k: PARAM_LABELS[k])
        if axis_z in (axis_x, axis_y):
            st.sidebar.warning("Axe Z identique à X ou Y : change-le pour une vraie 3D.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Plages des axes")
    ranges = {}

    def ui_range(param, default):
        dmin, dmax, dstep = default
        c = st.sidebar.container()
        c.markdown(f"**{PARAM_LABELS[param]}**")
        pmin = c.number_input(f"{param} min", value=dmin, key=param+"_min")
        pmax = c.number_input(f"{param} max", value=dmax, key=param+"_max")
        step = c.number_input(f"{param} pas", value=dstep, key=param+"_step")
        return float(pmin), float(pmax), float(step)

    ranges[axis_x] = ui_range(axis_x, DEFAULT_RANGES[axis_x])
    ranges[axis_y] = ui_range(axis_y, DEFAULT_RANGES[axis_y])
    if axis_z:
        ranges[axis_z] = ui_range(axis_z, DEFAULT_RANGES[axis_z])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres fixes")
    fixed_cfg = {
        "V_kV": None,
        "I_mA": None,
        "P_ubar": None,
        "Dch_cm": None,
        "flow_SCCM": None,
        "Dgrid_cm": None,
        "gas": gas,
        "mat_grid": mat_grid,
    }
    for p in NUMERIC_PARAMS:
        if p in (axis_x, axis_y) or p == axis_z:
            continue
        dmin, dmax, _ = DEFAULT_RANGES[p]
        default_value = 0.5 * (dmin + dmax)
        fixed_cfg[p] = st.sidebar.number_input(f"{PARAM_LABELS[p]} (fixe)", value=default_value)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Résumé")
        st.write(f"Gaz : **{gas}**")
        st.write(f"Matériau grille : **{mat_grid}**")
        st.write(f"Axe X : **{PARAM_LABELS[axis_x]}**")
        st.write(f"Axe Y : **{PARAM_LABELS[axis_y]}**")
        if axis_z:
            st.write(f"Axe Z : **{PARAM_LABELS[axis_z]}**")
        st.write(f"Grandeur affichée : **{grandeur_name}**")

    # ---------- MODE 2D ----------
    if mode == "Matrice 2D":
        x_vals, y_vals, M = build_matrix(axis_x, axis_y, ranges, fixed_cfg, qinfo["fn"])
        max_idx = np.unravel_index(np.argmax(M), M.shape)
        best_x = x_vals[max_idx[1]]
        best_y = y_vals[max_idx[0]]
        best_val = M[max_idx]

        with col_left:
            st.metric(qinfo["metric_label"], f"{best_val:.3e}")
            st.write(f"Optimum ≈ {PARAM_LABELS[axis_x]} = **{best_x:.3g}**")
            st.write(f"           {PARAM_LABELS[axis_y]} = **{best_y:.3g}**")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, origin="lower", aspect="equal", interpolation="nearest")
        ax.set_xlabel(PARAM_LABELS[axis_x])
        ax.set_ylabel(PARAM_LABELS[axis_y])
        ax.set_title(qinfo["title_2d"])
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=90)
        ax.set_yticklabels([f"{v:g}" for v in y_vals])
        ax.set_xlim(-0.5, len(x_vals) - 0.5)
        ax.set_ylim(-0.5, len(y_vals) - 0.5)
        ax.grid(color="k", linewidth=0.2)
        fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
        fig.colorbar(im, ax=ax, label=qinfo["colorbar_label"])
        with col_right:
            st.subheader("Matrice 2D")
            st.pyplot(fig)
        plt.close(fig)

    # ---------- MODE 3D ----------
    else:
        if axis_z is None:
            st.error("Choisis un axe Z pour activer l'empilement 3D.")
            return

        x_vals, y_vals, z_vals, M = build_matrix_stack(axis_x, axis_y, axis_z, ranges, fixed_cfg, qinfo["fn"])
        max_idx = np.unravel_index(np.argmax(M), M.shape)
        best_z = z_vals[max_idx[0]]
        best_y = y_vals[max_idx[1]]
        best_x = x_vals[max_idx[2]]
        best_val = M[max_idx]

        with col_left:
            st.metric(qinfo["metric_label"], f"{best_val:.3e}")
            st.write(f"Optimum ≈ {PARAM_LABELS[axis_x]} = **{best_x:.3g}**")
            st.write(f"           {PARAM_LABELS[axis_y]} = **{best_y:.3g}**")
            st.write(f"           {PARAM_LABELS[axis_z]} = **{best_z:.3g}**")

        # --- Empilement 3D ---
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        vmin, vmax = float(M.min()), float(M.max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
        gap = 0.4

        for k, z in enumerate(z_vals):
            z0 = k * (1.0 + gap)
            Z = np.full_like(X, z0)
            colors = cmap(norm(M[k]))
            ax.plot_surface(
                X, Y, Z,
                facecolors=colors,
                rstride=1, cstride=1,
                shade=False,
                edgecolor="k", linewidth=0.3
            )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.subplots_adjust(left=0.0, right=0.82, top=0.9, bottom=0.05)
        fig.colorbar(
            mappable,
            ax=ax,
            label=qinfo["colorbar_label"],
            shrink=0.7,
            pad=0.18
        )

        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=90)
        ax.set_yticklabels([f"{v:g}" for v in y_vals])
        ax.set_zticks([k * (1.0 + gap) for k in range(len(z_vals))])
        ax.set_zticklabels([f"{v:g}" for v in z_vals])
        ax.set_xlabel(PARAM_LABELS[axis_x])
        ax.set_ylabel(PARAM_LABELS[axis_y])
        ax.set_zlabel(PARAM_LABELS[axis_z])
        ax.set_title(qinfo["title_3d"])
        ax.view_init(elev=30, azim=-60)

        with col_right:
            st.subheader("Empilement 3D des matrices")
            st.pyplot(fig)
        plt.close(fig)

        # ---------- TENDANCES THÉORIQUES (uniquement si V/I/P/Y_n) ----------
        if (
            grandeur_name == "Production neutronique"
            and axis_x == "V_kV"
            and axis_y == "I_mA"
            and axis_z == "P_ubar"
        ):
            st.markdown("### Tendances théoriques")

            # --- Tendance 1D par tranche de pression ---
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            for k, P in enumerate(z_vals):
                layer = M[k]                      # shape (len(I), len(V))
                Y_mean = np.mean(layer, axis=0)   # moyenne sur I pour chaque V
                Y_mean = np.clip(Y_mean, 1e-40, None)
                logY = np.log10(Y_mean)
                # régression linéaire logY = a V + b
                coeff = np.polyfit(x_vals, logY, 1)
                V_fit = np.linspace(x_vals[0], x_vals[-1], 200)
                logY_fit = np.polyval(coeff, V_fit)
                Y_fit = 10**logY_fit

                ax2.scatter(x_vals, Y_mean, s=10, alpha=0.4)
                ax2.plot(V_fit, Y_fit, label=f"P = {P:.0f} µbar")

            ax2.set_yscale("log")
            ax2.set_xlabel("Tension cathode (kV)")
            ax2.set_ylabel("Y_th moyen (log)")
            ax2.set_title("Tendance théorique par tranche de pression")
            ax2.legend(fontsize=8)
            ax2.grid(True, which="both", linestyle=":")
            st.pyplot(fig2)
            plt.close(fig2)

            # --- Tendance 2D sur la surface (V, I) pour une pression choisie ---
            st.markdown("### Tendance 2D sur la surface (V, I) pour une pression donnée")
            idxP = st.slider(
                "Choisir l'indice de couche de pression",
                0, len(z_vals) - 1, len(z_vals) // 2
            )
            P_sel = z_vals[idxP]
            layer = M[idxP]  # shape (len(I), len(V))

            VV, II = np.meshgrid(x_vals, y_vals)  # V sur X, I sur Y
            Z_data = np.log10(np.clip(layer, 1e-40, None))

            X_design = np.column_stack([VV.ravel(), II.ravel(), np.ones(VV.size)])
            coeff2, _, _, _ = np.linalg.lstsq(X_design, Z_data.ravel(), rcond=None)
            a, b, c = coeff2
            Z_fit_flat = X_design @ coeff2
            Z_fit = Z_fit_flat.reshape(Z_data.shape)

            fig3 = plt.figure(figsize=(7, 5))
            ax3 = fig3.add_subplot(111, projection="3d")
            surf = ax3.plot_surface(VV, II, Z_data, cmap="viridis", alpha=0.8)
            ax3.plot_wireframe(VV, II, Z_fit, color="k", linewidth=0.7)
            fig3.colorbar(surf, shrink=0.6, pad=0.15, label="log10(Y_th)")
            ax3.set_xlabel("Tension cathode (kV)")
            ax3.set_ylabel("Courant cathode (mA)")
            ax3.set_zlabel("log10(Y_th)")
            ax3.set_title(f"Tendance 2D lissée à P = {P_sel:.0f} µbar")
            ax3.view_init(elev=30, azim=-60)
            st.pyplot(fig3)
            plt.close(fig3)

# --------- PAGE COMPARAISON ---------
def page_comparaison():
    st.title("Comparaison théorie / expérience (neutrons)")
    st.markdown("""
    Données expérimentales intégrées dans le code.
    On calcule Y_th pour chaque point, puis on affiche :
    \\[
    \\log_{10}\\left(\\frac{Y_{\\text{exp}}}{Y_{\\text{th}}}\\right)
    \\]
    """)

    experimental_data = [
        {"V_kV": 30, "I_mA": 7,  "P_ubar": 20, "Dch_cm": 20, "Dgrid_cm": 5, "neutrons_exp": 1.0e4},
        {"V_kV": 40, "I_mA": 10, "P_ubar": 15, "Dch_cm": 20, "Dgrid_cm": 5, "neutrons_exp": 1.0e5},
        {"V_kV": 45, "I_mA": 12, "P_ubar": 12, "Dch_cm": 20, "Dgrid_cm": 5, "neutrons_exp": 2.0e5},
        {"V_kV": 50, "I_mA": 15, "P_ubar": 10, "Dch_cm": 20, "Dgrid_cm": 5, "neutrons_exp": 1.0e7},
    ]
    df = pd.DataFrame(experimental_data)
    st.subheader("Données expérimentales (intégrées)")
    st.dataframe(df)

    st.sidebar.header("Options modèle (comparaison)")
    gas = st.sidebar.selectbox("Gaz (modèle)", ["D2", "H2"], index=0)

    df["flow_SCCM"] = 1.0
    df["mat_grid"] = "cuivre"
    df["gas"] = gas

    def row_to_cfg(row):
        return {
            "V_kV": float(row["V_kV"]),
            "I_mA": float(row["I_mA"]),
            "P_ubar": float(row["P_ubar"]),
            "Dch_cm": float(row["Dch_cm"]),
            "flow_SCCM": float(row["flow_SCCM"]),
            "Dgrid_cm": float(row["Dgrid_cm"]),
            "gas": row["gas"],
            "mat_grid": row["mat_grid"],
        }

    Y_th_list = []
    for _, r in df.iterrows():
        cfg = row_to_cfg(r)
        Y_th_list.append(fusion_rate_DD(cfg))
    df["Y_th"] = Y_th_list

    eps = 1e-30
    df["ratio"] = (df["neutrons_exp"] + eps) / (df["Y_th"] + eps)
    df["log10_ratio"] = np.log10(df["ratio"])

    st.subheader("Tableau avec modèle (Y_th) et ratio")
    st.dataframe(df[["V_kV", "I_mA", "P_ubar", "Dch_cm", "Dgrid_cm", "neutrons_exp", "Y_th", "log10_ratio"]])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Axes matrice de comparaison")
    possible_axes = ["V_kV", "I_mA", "P_ubar", "Dch_cm", "Dgrid_cm"]
    axis_x = st.sidebar.selectbox("Axe X", possible_axes, index=0)
    axis_y = st.sidebar.selectbox("Axe Y", possible_axes, index=1)

    pivot = df.pivot_table(
        index=axis_y,
        columns=axis_x,
        values="log10_ratio",
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    y_vals = pivot.index.values
    x_vals = pivot.columns.values
    M = pivot.values

    st.subheader("Matrice de comparaison")
    st.markdown(
        r"""
        Couleur = 
        \[
        \log_{10}\left(\frac{Y_{\text{exp}}}{Y_{\text{th}}}\right)
        \]
        0 → bon accord, >0 → exp > théorie, <0 → exp < théorie.
        """
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xlabel(axis_x)
    ax.set_ylabel(axis_y)
    ax.set_title(r"Matrice $\log_{10}(Y_{\text{exp}}/Y_{\text{th}})$")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=90)
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)
    ax.grid(color="k", linewidth=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}(Y_{\text{exp}}/Y_{\text{th}})$")
    st.pyplot(fig)
    plt.close(fig)

# --------- MAIN ---------
def main():
    st.set_page_config(page_title="Fusor Matrix Calculator", layout="wide")
    page = st.sidebar.radio("Mode global", ["Matrices théoriques", "Comparaison exp/théorie"])
    if page == "Matrices théoriques":
        page_theorie()
    else:
        page_comparaison()

if __name__ == "__main__":
    main()
