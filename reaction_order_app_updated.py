# reaction_order_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
from io import StringIO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Reaction Kinetics Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONSTANTS / HELPERS
# =========================================================

ORDER_INFO = {
    0: {
        "rate": r"\mathrm{Rate}=k",
        "integrated": r"[A]_t=[A]_0-kt",
        "plot": r"[A]\ \mathrm{vs}\ t",
        "half": r"t_{1/2}=\frac{[A]_0}{2k}",
    },
    1: {
        "rate": r"\mathrm{Rate}=k[A]",
        "integrated": r"\ln[A]_t=\ln[A]_0-kt",
        "plot": r"\ln[A]\ \mathrm{vs}\ t",
        "half": r"t_{1/2}=\frac{\ln 2}{k}",
    },
    2: {
        "rate": r"\mathrm{Rate}=k[A]^2",
        "integrated": r"\frac{1}{[A]_t}=\frac{1}{[A]_0}+kt",
        "plot": r"\frac{1}{[A]}\ \mathrm{vs}\ t",
        "half": r"t_{1/2}=\frac{1}{k[A]_0}",
    },
    3: {
        "rate": r"\mathrm{Rate}=k[A]^3",
        "integrated": r"\frac{1}{[A]_t^2}=\frac{1}{[A]_0^2}+2kt",
        "plot": r"\frac{1}{[A]^2}\ \mathrm{vs}\ t",
        "half": r"t_{1/2}=\frac{3}{2k[A]_0^2}",
    },
}

def fmt_num(x, sig=6):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{sig}g}"

def transformed_values(order, C):
    C = np.asarray(C, dtype=float)
    if order == 0:
        return C, "[A]"
    if order == 1:
        return np.log(C), "ln[A]"
    if order == 2:
        return 1 / C, "1/[A]"
    if order == 3:
        return 1 / (C ** 2), "1/[A]²"
    raise ValueError("Order must be 0, 1, 2, or 3.")

def analyze_integer_order(order, t, C):
    y, y_label = transformed_values(order, C)
    slope, intercept, r_value, p_value, std_err = linregress(t, y)

    if order == 0:
        k = -slope
        k_factor = 1.0
    elif order == 1:
        k = -slope
        k_factor = 1.0
    elif order == 2:
        k = slope
        k_factor = 1.0
    else:
        k = slope / 2
        k_factor = 2.0

    r2 = r_value ** 2
    k_positive = k > 0

    return {
        "order": order,
        "y": y,
        "y_label": y_label,
        "slope": slope,
        "intercept": intercept,
        "k": k,
        "R2": r2,
        "std_err": std_err,
        "valid_k": k_positive,
        "fit": intercept + slope * np.asarray(t),
    }

def general_n_r2(n, t, C):
    """R² for n != 1 using 1/C^(n-1) vs t."""
    if abs(n - 1) < 1e-7:
        return np.nan

    y = 1 / (np.asarray(C, dtype=float) ** (n - 1))
    return linregress(t, y).rvalue ** 2

def analyze_general_order(t, C, n_min, n_max):
    # For the continuous-order search, n=1 is represented by ln(C) vs t.
    def objective(n):
        if abs(n - 1) < 1e-5:
            y = np.log(C)
        else:
            y = 1 / (C ** (n - 1))
        return -linregress(t, y).rvalue ** 2

    # Avoid the singular point n=1 during scalar optimization.
    candidates = []

    left_max = min(n_max, 0.999)
    right_min = max(n_min, 1.001)

    if n_min < 1:
        res_left = minimize_scalar(
            objective,
            bounds=(n_min, left_max),
            method="bounded",
            options={"xatol": 1e-7},
        )
        candidates.append(res_left)

    if n_max > 1:
        res_right = minimize_scalar(
            objective,
            bounds=(right_min, n_max),
            method="bounded",
            options={"xatol": 1e-7},
        )
        candidates.append(res_right)

    # Explicitly test n=1.
    r2_one = linregress(t, np.log(C)).rvalue ** 2
    candidates.append(
        type("Result", (), {"x": 1.0, "fun": -r2_one})()
    )

    best = min(candidates, key=lambda x: x.fun)
    n_best = float(best.x)
    r2_best = -float(best.fun)

    # Calculate k from the appropriate transformed regression.
    if abs(n_best - 1) < 1e-5:
        y = np.log(C)
        slope, intercept, r_value, *_ = linregress(t, y)
        k = -slope
        y_label = "ln[A]"
        integrated = r"\ln[A]_t=\ln[A]_0-kt"
    else:
        y = 1 / (C ** (n_best - 1))
        slope, intercept, r_value, *_ = linregress(t, y)
        k = slope / (n_best - 1)
        y_label = r"1/[A]^{n-1}"
        integrated = (
            r"\frac{1}{[A]_t^{n-1}}-\frac{1}{[A]_0^{n-1}}"
            r"=(n-1)kt"
        )

    return {
        "n": n_best,
        "R2": r2_best,
        "k": k,
        "slope": slope,
        "intercept": intercept,
        "y": y,
        "y_label": y_label,
        "fit": intercept + slope * np.asarray(t),
        "integrated": integrated,
    }


def differential_rate_data(t, C):
    """Estimate -dC/dt from experimental C-vs-t data."""
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)

    if len(t) < 3:
        raise ValueError("At least 3 data points are required for differential analysis.")

    rate = -np.gradient(C, t)
    valid = np.isfinite(rate) & np.isfinite(C) & (rate > 0) & (C > 0)

    if valid.sum() < 3:
        raise ValueError(
            "Differential analysis needs at least 3 points with a positive "
            "estimated reaction rate (-dC/dt)."
        )

    return t[valid], C[valid], rate[valid], valid


def analyze_differential_order(t, C):
    """
    Differential method:
        -dC/dt = k C^n
        ln(-dC/dt) = ln(k) + n ln(C)
    """
    t_valid, C_valid, rate, valid = differential_rate_data(t, C)

    x = np.log(C_valid)
    y = np.log(rate)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "n": float(slope),
        "k": float(np.exp(intercept)),
        "R2": float(r_value ** 2),
        "slope": slope,
        "intercept": intercept,
        "x": x,
        "y": y,
        "rate": rate,
        "C": C_valid,
        "t": t_valid,
        "fit": intercept + slope * x,
        "valid_mask": valid,
    }


def half_life_for_order(order, k, C0):
    if not np.isfinite(k) or k <= 0:
        return np.nan
    if order == 0:
        return C0 / (2 * k)
    if order == 1:
        return np.log(2) / k
    if order == 2:
        return 1 / (k * C0)
    if order == 3:
        return 3 / (2 * k * C0**2)
    return np.nan

def general_half_life(n, k, C0):
    if not np.isfinite(k) or k <= 0:
        return np.nan
    if abs(n - 1) < 1e-5:
        return np.log(2) / k
    return (2**(n - 1) - 1) / ((n - 1) * k * C0**(n - 1))

def concentration_unit_factor(unit):
    # Convert entered concentration to M for kinetic calculations.
    if unit == "M":
        return 1.0
    if unit == "mol/L":
        return 1.0
    if unit == "mM":
        return 1e-3
    return 1.0

def time_unit_factor(unit):
    # Convert entered time to seconds for optional SI-style reporting.
    if unit == "s":
        return 1.0
    if unit == "min":
        return 60.0
    if unit == "h":
        return 3600.0
    return 1.0

def make_k_unit(order, c_unit, t_unit):
    if order == 0:
        return f"{c_unit}·{t_unit}⁻¹"
    if order == 1:
        return f"{t_unit}⁻¹"
    return f"{c_unit}^(1−{order:g})·{t_unit}⁻¹"

def make_general_k_unit(n, c_unit, t_unit):
    return f"{c_unit}^(1−n)·{t_unit}⁻¹"

# =========================================================
# HEADER
# =========================================================

st.title("⚗️ Reaction Kinetics Analyzer")
st.caption(
    "Determine reaction order, rate constant, half-life, and kinetic plots "
    "from experimental time–concentration data."
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    time_unit = st.selectbox(
        "Time unit",
        ["s", "min", "h"],
        help="Unit used for your entered time values."
    )

    concentration_unit = st.selectbox(
        "Concentration unit",
        ["M", "mol/L", "mM"],
        help="Unit used for your entered concentration values."
    )

    st.divider()

    st.subheader("Kinetic Method")

    method = st.radio(
        "Choose how to solve the C vs t data",
        ["Automatic (R²-guided)", "Integral Method", "Differential Method"],
        index=0,
        help=(
            "Automatic mode starts with the simpler integral method. "
            "If the integral fit is poor, the app recommends the differential method."
        ),
    )

    r2_threshold = st.slider(
        "Integral-method R² threshold",
        0.80, 0.99, 0.95, 0.01,
        help="Automatic mode recommends differential analysis when the best integral R² is below this value."
    )

    st.divider()

    st.subheader("General-order search")

    n_min = st.number_input(
        "Minimum n",
        min_value=-5.0,
        max_value=0.99,
        value=0.0,
        step=0.1
    )

    n_max = st.number_input(
        "Maximum n",
        min_value=1.01,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    st.caption(
        "The app tests n = 0, 1, 2, 3 and also searches continuously "
        "for the best-fitting n in the selected range."
    )

# =========================================================
# TABS
# =========================================================

tab_data, tab_plots, tab_analysis, tab_results = st.tabs(
    ["📊 Data Input", "📈 Plots", "🧪 Order Analysis", "📋 Results"]
)

# =========================================================
# DATA INPUT TAB
# =========================================================

with tab_data:
    st.header("Enter Experimental Data")

    input_method = st.radio(
        "Input method",
        ["Editable table", "Paste CSV", "Upload CSV / Excel"],
        horizontal=True
    )

    if input_method == "Editable table":
        default_data = pd.DataFrame({
            "Time": [0, 10, 20, 30, 40],
            "Concentration": [1.000, 0.819, 0.670, 0.549, 0.449],
        })

        data = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.NumberColumn(
                    "Time",
                    min_value=0,
                    format="%.6g"
                ),
                "Concentration": st.column_config.NumberColumn(
                    "Concentration",
                    min_value=0,
                    format="%.8g"
                ),
            }
        )

    elif input_method == "Paste CSV":
        csv_text = st.text_area(
            "Paste CSV data",
            value=(
                "Time,Concentration\n"
                "0,1.000\n"
                "10,0.819\n"
                "20,0.670\n"
                "30,0.549\n"
                "40,0.449\n"
            ),
            height=220,
        )

        try:
            data = pd.read_csv(StringIO(csv_text))
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

    else:
        uploaded = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"]
        )

        if uploaded is None:
            st.info("Upload a file to continue.")
            st.stop()

        try:
            if uploaded.name.lower().endswith(".csv"):
                data = pd.read_csv(uploaded)
            else:
                data = pd.read_excel(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

    # Normalize column names.
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]

    # Accept common alternative names.
    rename_map = {}
    for col in data.columns:
        low = col.lower().replace(" ", "")
        if low in {"t", "time", "time(s)", "times"}:
            rename_map[col] = "Time"
        elif low in {
            "c", "concentration", "concentration(m)",
            "[a]", "[a](m)", "conc"
        }:
            rename_map[col] = "Concentration"

    data = data.rename(columns=rename_map)

    if "Time" not in data.columns or "Concentration" not in data.columns:
        st.error(
            "Your data must contain columns named 'Time' and "
            "'Concentration' (or common equivalents such as t and C)."
        )
        st.stop()

    data = data[["Time", "Concentration"]].copy()
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data["Concentration"] = pd.to_numeric(
        data["Concentration"], errors="coerce"
    )

    original_count = len(data)

    # Remove invalid values.
    data = data.dropna()
    data = data[data["Time"] >= 0]
    data = data[data["Concentration"] > 0]
    data = data.sort_values("Time")
    data = data.drop_duplicates(subset="Time", keep="first")

    removed = original_count - len(data)

    if removed:
        st.warning(
            f"{removed} invalid/duplicate row(s) were removed. "
            "Concentration must be > 0 for logarithmic and inverse transforms."
        )

    if len(data) < 3:
        st.error("Please provide at least 3 valid data points.")
        st.stop()

    st.subheader("Cleaned Experimental Data")
    st.dataframe(data, use_container_width=True, hide_index=True)

    st.success(
        f"{len(data)} valid data points ready for kinetic analysis."
    )

# =========================================================
# CALCULATIONS
# =========================================================

t = data["Time"].to_numpy(dtype=float)
C = data["Concentration"].to_numpy(dtype=float)

integer_results = [
    analyze_integer_order(order, t, C)
    for order in [0, 1, 2, 3]
]

best_integer = max(integer_results, key=lambda r: r["R2"])

general_result = analyze_general_order(
    t, C, float(n_min), float(n_max)
)

differential_result = None
differential_error = None

try:
    differential_result = analyze_differential_order(t, C)
except Exception as exc:
    differential_error = str(exc)

integral_good_fit = best_integer["R2"] >= r2_threshold

if method == "Integral Method":
    selected_method = "Integral Method"
elif method == "Differential Method":
    selected_method = "Differential Method"
elif integral_good_fit:
    selected_method = "Integral Method"
else:
    selected_method = "Differential Method"

# =========================================================
# PLOTS TAB
# =========================================================

with tab_plots:
    st.header("📈 Kinetic Plots")

    # C vs t
    st.subheader("Concentration vs Time")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(t, C, label="Experimental data")
    ax.plot(t, C, linestyle="--", alpha=0.6)
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Concentration ({concentration_unit})")
    ax.set_title("Concentration vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # All integer order plots
    st.subheader("Comparison of 0th, 1st, 2nd and 3rd Order Fits")

    for result in integer_results:
        order = result["order"]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(
            t,
            result["y"],
            label="Experimental data"
        )
        ax.plot(
            t,
            result["fit"],
            linestyle="--",
            label="Linear regression"
        )
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(result["y_label"])
        ax.set_title(
            f"Order {order} | R² = {result['R2']:.6f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # Best general order plot
    st.subheader("Best Continuous-Order Fit")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        t,
        general_result["y"],
        label="Experimental data"
    )
    ax.plot(
        t,
        general_result["fit"],
        linestyle="--",
        label="Linear regression"
    )
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(general_result["y_label"])
    ax.set_title(
        f"Continuous order n = {general_result['n']:.4f} | "
        f"R² = {general_result['R2']:.6f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("Differential Method: ln(-dC/dt) vs ln(C)")

    if differential_result is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(
            differential_result["x"],
            differential_result["y"],
            label="Experimental rate data",
        )
        idx = np.argsort(differential_result["x"])
        ax.plot(
            differential_result["x"][idx],
            differential_result["fit"][idx],
            linestyle="--",
            label="Linear regression",
        )
        ax.set_xlabel("ln([A])")
        ax.set_ylabel("ln(-d[A]/dt)")
        ax.set_title(
            f"Differential Method | n = {differential_result['n']:.4f} | "
            f"R² = {differential_result['R2']:.6f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Differentiation can amplify experimental noise. Use the R²-guided "
            "recommendation together with experimental judgment."
        )
    else:
        st.warning(f"Differential method could not be evaluated: {differential_error}")

    st.divider()
    st.subheader("Differential Method Result")

    if differential_result is not None:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Differential order n", f"{differential_result['n']:.5f}")
        with d2:
            st.metric("Differential k", fmt_num(differential_result["k"]))
        with d3:
            st.metric("Differential R²", f"{differential_result['R2']:.6f}")

        st.latex(r"-\frac{d[A]}{dt}=k[A]^n")
        st.latex(r"\ln\left(-\frac{d[A]}{dt}\right)=\ln k+n\ln[A]")
    else:
        st.warning(f"Differential method unavailable: {differential_error}")

# =========================================================
# ORDER ANALYSIS TAB
# =========================================================

with tab_analysis:
    st.header("🧪 Reaction Order Analysis")

    st.subheader("Method Selection Guidance")

    if method == "Automatic (R²-guided)":
        if integral_good_fit:
            st.success(
                f"Start with the Integral Method. The best integer-order integral "
                f"fit has R² = {best_integer['R2']:.6f}, above the threshold "
                f"of {r2_threshold:.2f}. The Integral Method is recommended."
            )
        else:
            st.warning(
                f"The Integral Method was tested first, but its best integer-order "
                f"fit has R² = {best_integer['R2']:.6f}, below the threshold "
                f"of {r2_threshold:.2f}. Try the Differential Method for this "
                "dataset because the basic integral models do not describe it adequately."
            )
    elif method == "Integral Method":
        st.info(
            f"You selected the Integral Method. Best integer-order R² = "
            f"{best_integer['R2']:.6f}."
        )
        if not integral_good_fit:
            st.warning(
                f"R² is below {r2_threshold:.2f}. The Differential Method may "
                "be more appropriate for complicated kinetics."
            )
    else:
        if differential_result is not None:
            st.info(
                f"You selected the Differential Method: n = "
                f"{differential_result['n']:.4f}, R² = "
                f"{differential_result['R2']:.6f}."
            )
        else:
            st.error(f"Differential method unavailable: {differential_error}")

    st.subheader("Integer-Order Comparison")

    comparison_rows = []

    for r in integer_results:
        order = r["order"]
        half = half_life_for_order(
            order, r["k"], C[0]
        )

        comparison_rows.append({
            "Order": order,
            "k": r["k"],
            "R²": r["R2"],
            "Half-life": half,
            "Linear plot": ORDER_INFO[order]["plot"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df.style.format({
            "k": "{:.6g}",
            "R²": "{:.6f}",
            "Half-life": "{:.6g}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Best among the tested integer orders: "
        f"**{best_integer['order']}th order** "
        f"(R² = {best_integer['R2']:.6f})."
    )

    st.divider()

    st.subheader("Continuous Order Search")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Best continuous n",
            f"{general_result['n']:.5f}"
        )

    with c2:
        st.metric(
            "Best continuous R²",
            f"{general_result['R2']:.6f}"
        )

    st.write(
        "This searches for the order n that produces the most linear "
        "transformed concentration–time relationship."
    )

    if abs(general_result["n"] - 1) < 0.01:
        st.success("The continuous fit is essentially first order.")
    elif abs(general_result["n"] - round(general_result["n"])) < 0.03:
        st.success(
            f"The continuous fit is close to "
            f"{round(general_result['n'])}th order."
        )
    else:
        st.warning(
            "The best-fit order is fractional. This can be meaningful "
            "experimentally, but check your data quality and reaction model."
        )

# =========================================================
# RESULTS TAB
# =========================================================

with tab_results:
    st.header("📋 Final Results")

    C0 = C[0]

    if selected_method == "Differential Method" and differential_result is not None:
        n = differential_result["n"]
        k = differential_result["k"]
        r2 = differential_result["R2"]
        result_method_text = "Differential Method"
    else:
        n = general_result["n"]
        k = general_result["k"]
        r2 = general_result["R2"]
        result_method_text = "Integral Method"

    half = general_half_life(n, k, C0)

    st.info(f"**Method used for final result:** {result_method_text}")

    if abs(n - 1) < 1e-5:
        order_text = "1st order"
        rate_law_latex = r"\mathrm{Rate}=k[A]"
        k_unit = f"{time_unit}⁻¹"
        half_formula = r"t_{1/2}=\frac{\ln 2}{k}"
    else:
        order_text = f"{n:.4f} order"
        rate_law_latex = rf"\mathrm{{Rate}}=k[A]^{{{n:.4f}}}"
        k_unit = make_general_k_unit(n, concentration_unit, time_unit)
        half_formula = (
            rf"t_{{1/2}}="
            rf"\frac{{2^{{n-1}}-1}}{{(n-1)k[A]_0^{{n-1}}}}"
        )

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Reaction order", order_text)

    with b:
        st.metric("Rate constant k", fmt_num(k))

    with c:
        st.metric("R²", f"{r2:.6f}")

    with d:
        st.metric(
            "Half-life",
            f"{fmt_num(half)} {time_unit}"
        )

    st.divider()

    st.subheader("Differential Rate Law")
    st.latex(rate_law_latex)

    if selected_method == "Differential Method" and differential_result is not None:
        st.subheader("Differential Rate Law")
        st.latex(r"-\frac{d[A]}{dt}=k[A]^n")

        st.subheader("Differential Linearization")
        st.latex(r"\ln\left(-\frac{d[A]}{dt}\right)=\ln k+n\ln[A]")

        st.subheader("Linearized Plot")
        st.latex(r"\ln\left(-\frac{d[A]}{dt}\right)\ \mathrm{vs}\ \ln[A]")
    else:
        st.subheader("Integrated Rate Law")
        st.latex(general_result["integrated"])

        st.subheader("Linearized Plot")
        st.latex(
            rf"\frac{{1}}{{[A]^{{n-1}}}}\ \mathrm{{vs}}\ t"
            if abs(n - 1) >= 1e-5
            else r"\ln[A]\ \mathrm{{vs}}\ t"
        )

    st.subheader("Rate Constant Units")
    st.write(f"**{k_unit}**")

    st.subheader("Half-Life Equation")
    st.latex(half_formula)

    st.subheader("Initial Concentration")
    st.write(f"**[A]₀ = {fmt_num(C0)} {concentration_unit}**")

    st.divider()

    st.subheader("Integer-Order Recommendation")

    if best_integer["R2"] >= 0.99:
        recommendation = (
            f"The data has an excellent linear fit for "
            f"{best_integer['order']}th order among the tested integer orders."
        )
    else:
        recommendation = (
            f"The best integer-order fit is {best_integer['order']}th order, "
            f"but R² = {best_integer['R2']:.6f}. Consider experimental noise, "
            "additional reaction pathways, or testing a wider model."
        )

    st.write(recommendation)

    # Downloadable results
    result_rows = []

    for r in integer_results:
        result_rows.append({
            "Model": f"Order {r['order']}",
            "Order": r["order"],
            "k": r["k"],
            "R2": r["R2"],
            "Linearized variable": r["y_label"],
            "Half-life": half_life_for_order(
                r["order"], r["k"], C0
            ),
        })

    result_rows.append({
        "Model": "Continuous Integral",
        "Order": general_result["n"],
        "k": general_result["k"],
        "R2": general_result["R2"],
        "Linearized variable": general_result["y_label"],
        "Half-life": general_half_life(
            general_result["n"], general_result["k"], C0
        ),
    })

    if differential_result is not None:
        result_rows.append({
            "Model": "Differential",
            "Order": differential_result["n"],
            "k": differential_result["k"],
            "R2": differential_result["R2"],
            "Linearized variable": "ln(-d[A]/dt) vs ln[A]",
            "Half-life": general_half_life(
                differential_result["n"], differential_result["k"], C0
            ),
        })

    result_export = pd.DataFrame(result_rows)

    st.download_button(
        "⬇️ Download Analysis Results (CSV)",
        data=result_export.to_csv(index=False),
        file_name="reaction_kinetics_analysis.csv",
        mime="text/csv",
    )

    processed_data = data.copy()
    processed_data["ln(C)"] = np.log(C)
    processed_data["1/C"] = 1 / C
    processed_data["1/C^2"] = 1 / (C**2)

    st.download_button(
        "⬇️ Download Transformed Data (CSV)",
        data=processed_data.to_csv(index=False),
        file_name="reaction_kinetics_transformed_data.csv",
        mime="text/csv",
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "Reaction Kinetics Analyzer • Integral and Differential kinetic methods "
    "with R²-guided method selection."
)
