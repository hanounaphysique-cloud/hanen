import numpy as np
import streamlit as st
import pandas as pd

# ============================
#  CONSTANTES PHYSIQUES
# ============================
MU0 = 4e-7 * np.pi  # perméabilité du vide [H/m]
UBAR_TO_PA = 0.1    # 1 µbar = 0.1 Pa


# ============================
#  OUTILS COMMUNS RENDEMENT
# ============================

def compute_energy_chain(E_in_kJ, stages):
    """
    Calcule la chaîne d'énergie pour une configuration.
    stages = liste de dicts:
      {"name": str, "eta": float (0-1), "recoverable": bool}
    Retourne un DataFrame et un résumé.
    """
    rows = []
    E_in = E_in_kJ
    total_recov = 0.0

    for s in stages:
        E_out = E_in * s["eta"]
        loss = E_in - E_out
        recov = loss if s["recoverable"] else 0.0
        total_recov += recov

        rows.append({
            "Étape": s["name"],
            "Énergie entrante (kJ)": E_in,
            "Rendement étape (%)": s["eta"] * 100.0,
            "Énergie sortante (kJ)": E_out,
            "Pertes (kJ)": loss,
            "Pertes récupérables (kJ)": recov,
        })

        E_in = E_out

    df = pd.DataFrame(rows)
    useful = E_in
    total_loss = E_in_kJ - useful
    non_recov = total_loss - total_recov
    eta_global = useful / E_in_kJ if E_in_kJ > 0 else 0.0

    summary = {
        "E_entrée (kJ)": E_in_kJ,
        "E_utile finale (kJ)": useful,
        "Rendement global (%)": eta_global * 100.0,
        "Pertes totales (kJ)": total_loss,
        "Pertes récupérables (kJ)": total_recov,
        "Pertes non récupérables (kJ)": non_recov,
        # CECI = conversion élec par kJ
        "Conversion élec par kJ (kJ_out/kJ_in)": eta_global,
    }
    return df, summary


# ============================
#  1) PAGE RENDEMENT LIBRE
# ============================

def rendement_page():
    st.title("1) Rendement des configurations (référencé à 1 kJ)")

    st.markdown(
        """
        On suppose **1 kJ** d'énergie électrique entrante (modifiable ci-dessous).  
        Pour chaque configuration, on définit une **chaîne d'étapes** avec un rendement
        et on indique si les pertes de l'étape sont **récupérables** ou non.

        👉 **Conversion élec par kJ = énergie utile par kJ d'électricité entrante = kJ_out / kJ_in.**
        """
    )

    E_in_kJ = st.number_input("Énergie électrique entrante de référence (kJ)", 0.1, 10000.0, 1.0)

    config_names = ["Farnsworth", "Polywell", "Z-pinch", "Tokamak"]

    results = []

    for cfg_name in config_names:
        with st.expander(f"Configuration : {cfg_name}", expanded=(cfg_name == "Farnsworth")):
            st.write("Définis les étapes de la chaîne énergétique.")
            n_stages = st.slider(f"Nombre d'étapes pour {cfg_name}", 1, 6, 3, key=f"n_{cfg_name}")

            stages = []
            for i in range(n_stages):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    name = st.text_input(
                        f"Nom étape {i+1}",
                        value=f"Étape {i+1}",
                        key=f"name_{cfg_name}_{i}",
                    )
                with col2:
                    eta_pct = st.number_input(
                        f"Rendement {i+1} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=90.0,
                        key=f"eta_{cfg_name}_{i}",
                    )
                with col3:
                    recov = st.checkbox(
                        f"Pertes récupérables ?",
                        value=(i == 0),
                        key=f"recov_{cfg_name}_{i}",
                    )

                stages.append({
                    "name": name,
                    "eta": eta_pct / 100.0,
                    "recoverable": recov,
                })

            df, summary = compute_energy_chain(E_in_kJ, stages)
            st.markdown("**Tableau détaillé (ramené à 1 kJ d'entrée)**")
            st.dataframe(df.style.format({
                "Énergie entrante (kJ)": "{:.3f}",
                "Énergie sortante (kJ)": "{:.3f}",
                "Pertes (kJ)": "{:.3f}",
                "Pertes récupérables (kJ)": "{:.3f}",
                "Rendement étape (%)": "{:.1f}",
            }))

            st.markdown("**Résumé de la configuration**")
            colA, colB, colC = st.columns(3)
            colA.metric("Rendement global", f"{summary['Rendement global (%)']:.2f} %")
            colB.metric("Énergie utile finale", f"{summary['E_utile finale (kJ)']:.3f} kJ")
            colC.metric(
                "Conversion élec par kJ (kJ_out/kJ_in)",
                f"{summary['Conversion élec par kJ (kJ_out/kJ_in)']:.3f}",
            )

            st.write(
                f"Pertes totales : **{summary['Pertes totales (kJ)']:.3f} kJ** "
                f"dont récupérables **{summary['Pertes récupérables (kJ)']:.3f} kJ** "
                f"et non récupérables **{summary['Pertes non récupérables (kJ)']:.3f} kJ**."
            )

            results.append({"Configuration": cfg_name, **summary})

    st.markdown("---")
    st.subheader("Comparaison synthétique des configurations (par kJ d'entrée)")

    if results:
        df_res = pd.DataFrame(results)
        st.dataframe(
            df_res[
                [
                    "Configuration",
                    "Rendement global (%)",
                    "Conversion élec par kJ (kJ_out/kJ_in)",
                    "Pertes totales (kJ)",
                    "Pertes récupérables (kJ)",
                    "Pertes non récupérables (kJ)",
                ]
            ].style.format(
                {
                    "Rendement global (%)": "{:.2f}",
                    "Conversion élec par kJ (kJ_out/kJ_in)": "{:.3f}",
                    "Pertes totales (kJ)": "{:.3f}",
                    "Pertes récupérables (kJ)": "{:.3f}",
                    "Pertes non récupérables (kJ)": "{:.3f}",
                }
            )
        )

    # Petit convertisseur kJ ↔ kWh
    st.markdown("---")
    st.subheader("Convertisseur kJ ↔ kWh (centrale électrique)")

    col1, col2 = st.columns(2)
    with col1:
        E_kJ = st.number_input("Énergie (kJ)", min_value=0.0, value=3600.0, key="conv_kJ")
        st.write(f"{E_kJ:.3f} kJ = {E_kJ/3600:.6f} kWh")
    with col2:
        E_kWh = st.number_input("Énergie (kWh)", min_value=0.0, value=1.0, key="conv_kWh")
        st.write(f"{E_kWh:.6f} kWh = {E_kWh*3600:.3f} kJ")


# ============================
#  2) PAGE Z-PINCH IDÉAL
# ============================

def zpinch_current_from_pressure(P_ubar, radius_cm):
    """
    Courant nécessaire pour équilibre pression magnétique = pression gaz.
    P en µbar, rayon en cm. Retourne I en ampères.
    I = sqrt(8*pi^2 * r^2 * P / mu0)
    """
    P_pa = P_ubar * UBAR_TO_PA  # Pa
    r_m = radius_cm / 100.0
    I = np.sqrt(8.0 * np.pi**2 * r_m**2 * P_pa / MU0)
    return I


def zpinch_voltage_for_energy(E_kJ, I_A, pulse_ns):
    """
    Tension nécessaire pour délivrer E_kJ en une impulsion de pulse_ns,
    à courant I_A supposé constant sur l'impulsion.
    V = E / (I * dt)
    """
    E_J = E_kJ * 1e3
    dt = pulse_ns * 1e-9
    if I_A <= 0 or dt <= 0:
        return np.nan
    V = E_J / (I_A * dt)
    return V


def zpinch_page():
    st.title("2) Z-pinch idéal (tension et courant vs pression, taille)")

    st.markdown(
        """
        Modèle de **Z-pinch idéal sans pertes** :

        * équilibre : pression magnétique = pression du gaz,  
        * impulsion rectangulaire de durée Δt,  
        * toute l'énergie électrique **E** est déposée dans le plasma :
          \\(E = V I \\Delta t\\).
        """
    )

    colE, colT = st.columns(2)
    with colE:
        E_kJ = st.number_input("Énergie par impulsion (kJ)", 0.01, 10000.0, 1.0)
    with colT:
        pulse_ns = st.number_input("Durée d'impulsion Δt (ns)", 1.0, 1000.0, 100.0)

    st.sidebar.header("Paramètres Z-pinch")
    default_pressures = [50.0, 150.0]
    pressures = st.sidebar.text_input(
        "Pressions (µbar), séparées par des virgules",
        value=", ".join(str(p) for p in default_pressures),
    )
    try:
        P_list = [float(p.strip()) for p in pressures.split(",") if p.strip()]
    except ValueError:
        P_list = default_pressures

    D_min = st.sidebar.number_input("Diamètre min (cm)", 0.1, 100.0, 1.0)
    D_max = st.sidebar.number_input("Diamètre max (cm)", 0.1, 200.0, 10.0)
    D_step = st.sidebar.number_input("Pas en diamètre (cm)", 0.1, 50.0, 1.0)

    diameters = np.arange(D_min, D_max + 1e-9, D_step)

    rows = []
    for P_ubar in P_list:
        for D_cm in diameters:
            radius_cm = D_cm / 2.0
            I_A = zpinch_current_from_pressure(P_ubar, radius_cm)
            V_V = zpinch_voltage_for_energy(E_kJ, I_A, pulse_ns)
            rows.append({
                "Pression (µbar)": P_ubar,
                "Diamètre (cm)": D_cm,
                "Courant I (MA)": I_A / 1e6,
                "Tension V (kV)": V_V / 1e3,
                "Puissance instantanée (GW)": (V_V * I_A) / 1e9,
            })

    df = pd.DataFrame(rows)

    st.subheader("Résultats (modèle idéal, pas de pertes)")
    st.dataframe(
        df.sort_values(["Pression (µbar)", "Diamètre (cm)"]).style.format(
            {
                "Diamètre (cm)": "{:.2f}",
                "Courant I (MA)": "{:.3f}",
                "Tension V (kV)": "{:.3f}",
                "Puissance instantanée (GW)": "{:.3f}",
            }
        )
    )

    st.subheader("Tendances graphiques")

    try:
        import matplotlib.pyplot as plt

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Courant vs diamètre**")
            fig1, ax1 = plt.subplots()
            for P in sorted(set(df["Pression (µbar)"])):
                sub = df[df["Pression (µbar)"] == P]
                ax1.plot(sub["Diamètre (cm)"], sub["Courant I (MA)"], marker="o", label=f"{P:.0f} µbar")
            ax1.set_xlabel("Diamètre (cm)")
            ax1.set_ylabel("Courant I (MA)")
            ax1.grid(True, linestyle=":")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.markdown("**Tension vs diamètre**")
            fig2, ax2 = plt.subplots()
            for P in sorted(set(df["Pression (µbar)"])):
                sub = df[df["Pression (µbar)"] == P]
                ax2.plot(sub["Diamètre (cm)"], sub["Tension V (kV)"], marker="o", label=f"{P:.0f} µbar")
            ax2.set_xlabel("Diamètre (cm)")
            ax2.set_ylabel("Tension V (kV)")
            ax2.grid(True, linestyle=":")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as exc:
        st.info(f"Impossible de tracer les graphes (matplotlib non disponible ?) : {exc}")


# ============================
#  3) PAGE SCÉNARIOS IDÉAUX FUSION
# ============================

IDEAL_SCENARIOS = {
    "Tokamak": [
        {
            "name": "Systèmes électriques → champs / chauffage",
            "eta": 0.95,
            "recoverable": True,
        },
        {
            "name": "Couplage vers le plasma",
            "eta": 0.90,
            "recoverable": False,
        },
        {
            "name": "Conversion chaleur → électricité",
            "eta": 0.45,
            "recoverable": True,
        },
    ],
    "Polywell": [
        {
            "name": "Électronique & bobines",
            "eta": 0.95,
            "recoverable": True,
        },
        {
            "name": "Couplage vers le plasma",
            "eta": 0.95,
            "recoverable": False,
        },
        {
            "name": "Conversion directe ions → électricité",
            "eta": 0.90,
            "recoverable": True,
        },
    ],
    "Z-pinch": [
        {
            "name": "Chaîne pulse-power",
            "eta": 0.90,
            "recoverable": True,
        },
        {
            "name": "Couplage vers le pinch",
            "eta": 0.90,
            "recoverable": False,
        },
        {
            "name": "Conversion chaleur → électricité",
            "eta": 0.45,
            "recoverable": True,
        },
    ],
    "Farnsworth": [
        {
            "name": "Alimentation & électronique",
            "eta": 0.90,
            "recoverable": True,
        },
        {
            "name": "Confinement / grilles",
            "eta": 0.20,
            "recoverable": False,
        },
        {
            "name": "Conversion chaleur → électricité",
            "eta": 0.45,
            "recoverable": True,
        },
    ],
}

# Centrales classiques : charbon, gaz, fission
# Rendements globaux typiques
CLASSIC_PLANTS = [
    {"Technologie": "Charbon (supercritique)", "eta": 0.38},
    {"Technologie": "Gaz (cycle combiné)", "eta": 0.55},
    {"Technologie": "Fission (PWR)", "eta": 0.33},
]


def ideal_scenarios_page():
    st.title("3) Scénarios idéaux fusion + centrales classiques")

    st.markdown(
        """
        **Partie 1 : concepts de fusion (scénarios idéaux hypothétiques)**  
        Rendements très optimistes → bornes hautes pédagogiques.

        👉 La colonne clé est **“Conversion élec par kJ (kJ_out/kJ_in)”** :  
        combien de kJ d'électricité on récupère pour 1 kJ d'électricité consommée.
        """
    )

    E_in_kJ = st.number_input("Énergie électrique entrante de référence (kJ)", 0.1, 10000.0, 1.0)

    all_results = []

    for cfg_name, stages in IDEAL_SCENARIOS.items():
        st.markdown(f"---\n### {cfg_name} — scénario idéal (fusion)")

        df, summary = compute_energy_chain(E_in_kJ, stages)

        st.markdown("**Étapes et bilan détaillé**")
        st.dataframe(df.style.format({
            "Énergie entrante (kJ)": "{:.3f}",
            "Énergie sortante (kJ)": "{:.3f}",
            "Pertes (kJ)": "{:.3f}",
            "Pertes récupérables (kJ)": "{:.3f}",
            "Rendement étape (%)": "{:.1f}",
        }))

        colA, colB, colC = st.columns(3)
        colA.metric("Rendement global", f"{summary['Rendement global (%)']:.2f} %")
        colB.metric("Énergie utile finale", f"{summary['E_utile finale (kJ)']:.3f} kJ")
        colC.metric(
            "Conversion élec par kJ (kJ_out/kJ_in)",
            f"{summary['Conversion élec par kJ (kJ_out/kJ_in)']:.3f}",
        )

        st.write(
            f"Pertes totales : **{summary['Pertes totales (kJ)']:.3f} kJ** "
            f"dont récupérables **{summary['Pertes récupérables (kJ)']:.3f} kJ** "
            f"et non récupérables **{summary['Pertes non récupérables (kJ)']:.3f} kJ**."
        )

        all_results.append({"Configuration": cfg_name, **summary})

    st.markdown("---")
    st.subheader("Comparaison globale des scénarios fusion (par kJ d'entrée)")

    if all_results:
        df_res = pd.DataFrame(all_results)
        st.dataframe(
            df_res[
                [
                    "Configuration",
                    "Rendement global (%)",
                    "Conversion élec par kJ (kJ_out/kJ_in)",
                    "Pertes totales (kJ)",
                    "Pertes récupérables (kJ)",
                    "Pertes non récupérables (kJ)",
                ]
            ].style.format(
                {
                    "Rendement global (%)": "{:.2f}",
                    "Conversion élec par kJ (kJ_out/kJ_in)": "{:.3f}",
                    "Pertes totales (kJ)": "{:.3f}",
                    "Pertes récupérables (kJ)": "{:.3f}",
                    "Pertes non récupérables (kJ)": "{:.3f}",
                }
            )
        )

    # ====== PARTIE 2 : Centrales classiques (une ligne par techno) ======
    st.markdown("---")
    st.subheader("Centrales classiques : une ligne par technologie")

    # 1 Wh = 3,6 kJ → kJ_in / Wh_out = 3,6 / eta
    rows_classic = []
    for plant in CLASSIC_PLANTS:
        eta = plant["eta"]
        kJ_per_Wh = 3.6 / eta if eta > 0 else np.nan
        rows_classic.append({
            "Technologie": plant["Technologie"],
            "Rendement (%)": eta * 100.0,
            "Conversion élec par kJ (kJ_out/kJ_in)": eta,
            "kJ entrant / Wh sortant": kJ_per_Wh,
        })

    df_classic = pd.DataFrame(rows_classic)
    st.dataframe(
        df_classic.style.format({
            "Rendement (%)": "{:.1f}",
            "Conversion élec par kJ (kJ_out/kJ_in)": "{:.3f}",
            "kJ entrant / Wh sortant": "{:.2f}",
        })
    )

    st.markdown(
        """
        👉 **Charbon, gaz, fission** sont donc exprimés exactement comme les concepts de fusion :  
        - une **conversion élec par kJ (kJ_out/kJ_in)**  
        - un **rendement (%)**  
        - et le ratio **kJ entrant / Wh sortant** sur **une seule ligne par technologie**.
        """
    )

    # ====== PARTIE 3 : Centrale électrique complète (MWh in → MWh out) ======
    st.markdown("---")
    st.subheader("Interprétation centrale électrique (MWh entrants → MWh sortants)")

    colP, colT = st.columns(2)
    with colP:
        P_in_MW = st.number_input(
            "Puissance électrique entrante de la centrale (MW)",
            min_value=0.0,
            value=20.0,  # ex : 20 MW
            key="plant_P_in_MW",
        )
    with colT:
        t_h = st.number_input(
            "Durée de fonctionnement (heures)",
            min_value=0.0,
            value=1.0,
            key="plant_t_h",
        )

    E_in_MWh_plant = P_in_MW * t_h
    E_in_kWh_plant = E_in_MWh_plant * 1000.0

    st.write(
        f"**Entrée centrale** : {P_in_MW:.3f} MW pendant {t_h:.3f} h "
        f"→ **{E_in_MWh_plant:.3f} MWh** consommés "
        f"(soit {E_in_kWh_plant*3600:.0f} kJ)."
    )

    if all_results and E_in_MWh_plant > 0:
        rows_conv = []
        for res in all_results:
            cfg = res["Configuration"]
            eta = res["Conversion élec par kJ (kJ_out/kJ_in)"]  # = kJ_out/kJ_in = kWh_out/kWh_in
            E_out_MWh = eta * E_in_MWh_plant
            rows_conv.append({
                "Configuration": cfg,
                "Conversion élec par kJ (kJ_out/kJ_in)": eta,
                "E_in (MWh)": E_in_MWh_plant,
                "E_out (MWh)": E_out_MWh,
            })

        df_conv = pd.DataFrame(rows_conv)
        st.dataframe(
            df_conv.style.format({
                "Conversion élec par kJ (kJ_out/kJ_in)": "{:.3f}",
                "E_in (MWh)": "{:.3f}",
                "E_out (MWh)": "{:.3f}",
            })
        )

        st.markdown(
            """
            👉 Pour chaque scénario de fusion, la colonne **“Conversion élec par kJ (kJ_out/kJ_in)”**  
            est exactement le même ratio que **MWh sortants / MWh entrants** à l’échelle de la centrale.
            """
        )


# ============================
#  MAIN
# ============================

def main():
    st.set_page_config(page_title="Calculatrice rendement & Z-pinch", layout="wide")

    page = st.sidebar.radio(
        "Choisir le mode",
        [
            "Rendement configurations (1 kJ)",
            "Z-pinch idéal (tension / courant)",
            "Scénarios idéaux (comparaison)",
        ],
    )

    if page.startswith("Rendement"):
        rendement_page()
    elif page.startswith("Z-pinch"):
        zpinch_page()
    else:
        ideal_scenarios_page()


if __name__ == "__main__":
    main()
