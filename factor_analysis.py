"""
Factor analysis "by hand" (principal components + pairwise varimax rotation).

numpy is used only for basic linear algebra (eigh, inv, matrix products).
NO sklearn, factor_analyzer, statsmodels or any other FA/PCA/varimax library.

Run:     python factor_analysis.py
Output:  console + resultados.txt
"""

import itertools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DDOF = 1            # degrees of freedom for the standard deviation (n-1)
DEC = 4             # decimals to print
ITERAR = False      # False = a single sweep over the pairs; True = iterate until convergence
MAX_CICLOS = 100    # only used if ITERAR = True
TOL = 1e-8          # convergence tolerance (angles) if ITERAR = True
ARCHIVO_SALIDA = "resultados.txt"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.{DEC}f}")

# ---------------------------------------------------------------------------
# Printing utilities (everything printed is also stored for the .txt)
# ---------------------------------------------------------------------------
_LINEAS = []


def out(texto=""):
    """Print to the console and store the line for resultados.txt."""
    texto = str(texto)
    print(texto)
    _LINEAS.append(texto)


def titulo(texto):
    out()
    out("=" * 78)
    out(texto)
    out("=" * 78)


def mostrar(nombre, matriz, filas=None, columnas=None):
    """Print a matrix/vector ALWAYS with a title, rounded to DEC decimals."""
    out(f"\n--- {nombre} ---")
    arr = np.asarray(matriz, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        filas = filas if filas is not None else [""]
    df = pd.DataFrame(np.round(arr, DEC), index=filas, columns=columns_or(columnas, arr))
    out(df.to_string())


def columns_or(columnas, arr):
    return columnas if columnas is not None else [f"c{j + 1}" for j in range(arr.shape[1])]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def cargar_datos():
    """Return X (10x6) as a DataFrame; the Student column is only the index."""
    datos = [
        [1, 85.5, 88.0, 42.1, 45.0, 78.2, 80.5],
        [2, 40.2, 42.5, 91.0, 93.5, 50.1, 48.0],
        [3, 92.0, 90.4, 88.5, 89.0, 95.4, 94.1],
        [4, 35.0, 38.2, 33.1, 35.4, 32.0, 30.5],
        [5, 78.1, 80.0, 45.2, 42.1, 82.4, 85.0],
        [6, 45.0, 43.2, 85.4, 88.0, 40.1, 42.3],
        [7, 89.5, 91.2, 90.1, 92.4, 91.0, 89.5],
        [8, 31.2, 30.0, 88.4, 85.1, 33.5, 35.0],
        [9, 82.0, 85.4, 38.0, 40.2, 80.1, 81.4],
        [10, 50.4, 48.1, 92.5, 94.0, 52.3, 50.1],
    ]
    cols = ["Student", "Y1_Alg1", "Y2_Alg2", "Y3_Log1", "Y4_Log2", "Y5_Mod1", "Y6_Mod2"]
    df = pd.DataFrame(datos, columns=cols).set_index("Student")
    df.index = [f"Estudiante {i}" for i in df.index]
    return df


# ---------------------------------------------------------------------------
# Step 1. Standard deviation and standardization
# ---------------------------------------------------------------------------
def paso1_estandarizar(X, variables, estudiantes):
    titulo("PASO 1. Medias, desviaciones estándar y matriz estandarizada Z")
    n = X.shape[0]
    media = X.sum(axis=0) / n
    # S_i = sqrt( Σ (x - media)^2 / (n - DDOF) )
    S = np.sqrt(((X - media) ** 2).sum(axis=0) / (n - DDOF))
    Z = (X - media) / S
    out(f"n = {n} estudiantes, p = {X.shape[1]} variables, DDOF = {DDOF}")
    mostrar("Medias de cada variable", media, ["Media"], variables)
    mostrar("Desviaciones estándar S_i", S, ["S_i"], variables)
    mostrar("Matriz estandarizada Z = (X - media) / S_i  (10x6)", Z, estudiantes, variables)
    return Z, media, S


# ---------------------------------------------------------------------------
# Step 2. Correlation matrix
# ---------------------------------------------------------------------------
def paso2_correlacion(Z, variables):
    titulo("PASO 2. Matriz de correlación R = (1/(n-1)) Z^T Z")
    n = Z.shape[0]
    R = (Z.T @ Z) / (n - 1)
    mostrar("Matriz de correlación R (6x6)", R, variables, variables)
    diag = np.diag(R)
    mostrar("Diagonal de R (debe ser 1)", diag, ["diag(R)"], variables)
    # Note: the diagonal is exactly 1 only if DDOF = 1 (consistent with the 1/(n-1)).
    out(f"¿Diagonal = 1?  {np.allclose(diag, 1.0)}")
    return R


# ---------------------------------------------------------------------------
# Step 3. Spectral decomposition + Kaiser criterion
# ---------------------------------------------------------------------------
def paso3_espectral(R, variables):
    titulo("PASO 3. Descomposición espectral R = V Λ V^T")
    lam, V = np.linalg.eigh(R)            # eigh returns them in ascending order
    orden = np.argsort(lam)[::-1]         # largest to smallest
    lam, V = lam[orden], V[:, orden]

    # Sign convention (eigenvectors are only defined up to sign):
    # make each eigenvector sum to a positive value so results are reproducible.
    signos = np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    V = V * signos

    p = len(lam)
    etiquetas = [f"λ{k + 1}" for k in range(p)]
    var_exp = 100 * lam / lam.sum()
    var_acum = np.cumsum(var_exp)

    mostrar("Eigenvalores λ (orden descendente)", lam, ["λ"], etiquetas)
    mostrar("Eigenvectores V (columna k ↔ λk)", V, variables, [f"v{k + 1}" for k in range(p)])
    mostrar("Varianza explicada (%)",
            np.vstack([var_exp, var_acum]), ["% explicada", "% acumulada"], etiquetas)

    # Kaiser criterion: since R is a correlation matrix, each standardized variable
    # contributes variance 1. A factor with λ <= 1 explains less variance than a SINGLE
    # original variable, so it does not summarize information and is discarded.
    m = int(np.sum(lam > 1))
    out(f"\nCriterio de Kaiser (λ > 1): se conservan m = {m} factores.")
    for k in range(p):
        estado = "SE CONSERVA" if lam[k] > 1 else "se descarta (explica menos que una variable)"
        out(f"  λ{k + 1} = {lam[k]:.{DEC}f}  -> {estado}")
    out(f"Varianza total explicada por los {m} factores: {var_acum[m - 1]:.{DEC}f} %")
    return lam, V, m


# ---------------------------------------------------------------------------
# Step 4. Unrotated loadings
# ---------------------------------------------------------------------------
def paso4_loadings(lam, V, m, variables):
    titulo("PASO 4. Matriz de loadings sin rotar L = V_m · sqrt(Λ_m)")
    L = V[:, :m] * np.sqrt(lam[:m])
    h2 = (L ** 2).sum(axis=1)
    factores = [f"F{k + 1}" for k in range(m)]
    mostrar(f"Loadings sin rotar L (6x{m})", L, variables, factores)
    mostrar("Comunalidades h_i² = Σ_k L_ik²", h2, ["h_i²"], variables)
    return L, h2


# ---------------------------------------------------------------------------
# Step 5. Explanation of the general factor G (text only)
# ---------------------------------------------------------------------------
def paso5_factor_g(L, variables):
    titulo("PASO 5. ¿Por qué la solución sin rotar no aísla la especialidad? (factor G)")
    # In the unrotated solution, F1 is the direction of maximum common variance. Since
    # the subjects are positively correlated, F1 loads strongly on ALL variables: it is
    # a general factor "g" (overall performance). A student's F1 score mixes Algebra,
    # Logic and Modeling, so it reflects their general level, not their specialty. The
    # following factors are bipolar (+/-) contrasts that are hard to interpret.
    # Moreover, the solution is NOT unique: for any orthogonal matrix T,
    # (L T)(L T)^T = L T T^T L^T = L L^T, i.e. every rotation reproduces the same R.
    # That is why we rotate: among the infinitely many equivalent solutions we pick the
    # one with "simple structure" (each variable loads high on a single factor).
    texto = """\
In the unrotated space, F1 loads strongly on ALL variables: it is a general
factor 'g' (overall performance). Each student's F1 score is dominated by their
general level, not by their specialty (Algebra/Logic/Modeling), and the other
factors are bipolar (+/-) contrasts that are hard to interpret.
Moreover, the solution is not unique: for any orthogonal T,
(LT)(LT)^T = L T T^T L^T = L L^T, so every rotation reproduces the same R.
That is why we rotate toward a simple structure."""
    out(texto)
    out("\nCargas de F1 (sin rotar) en cada variable:")
    for v, c in zip(variables, L[:, 0]):
        out(f"  {v:8s}: {c: .{DEC}f}")
    altas = [v for v, c in zip(variables, L[:, 0]) if abs(c) >= 0.5]
    out(f"Mínimo |carga| de F1 = {np.abs(L[:, 0]).min():.{DEC}f}, "
        f"promedio = {np.abs(L[:, 0]).mean():.{DEC}f}")
    out(f"F1 tiene carga distinta de cero en las {len(variables)} variables "
        f"(|carga| >= 0.5 en {len(altas)} de ellas: {', '.join(altas)}).")
    out("Ninguna variable queda 'limpia' en un solo factor: cada fila de L mezcla F1 y F2,")
    out("por eso el puntaje sin rotar no separa la especialidad del nivel general.")


# ---------------------------------------------------------------------------
# Step 6. Row normalization (Kaiser normalization)
# ---------------------------------------------------------------------------
def paso6_normalizar(L, h2, variables):
    titulo("PASO 6. Normalización por filas L_norm[i,:] = L[i,:] / h_i")
    h = np.sqrt(h2)                      # stored to denormalize in step 8
    L_norm = L / h[:, None]
    m = L.shape[1]
    mostrar("h_i = sqrt(h_i²) (guardados para desnormalizar)", h, ["h_i"], variables)
    mostrar("L_norm (cada fila con norma 1)", L_norm, variables, [f"F{k + 1}" for k in range(m)])
    return L_norm, h


# ---------------------------------------------------------------------------
# Step 7. Pairwise varimax rotation of columns
# ---------------------------------------------------------------------------
def angulo_optimo(A, B, C, D, p):
    """
    Optimal rotation angle for a pair of columns (formula from class).
        4θ = arctan2( D - 2AB/p , C - (A² - B²)/p )
    If your version of the formula differs, change it ONLY here.
    """
    num = D - 2 * A * B / p
    den = C - (A ** 2 - B ** 2) / p
    return 0.25 * np.arctan2(num, den)


def rotar_par(L, i, j):
    """Compute A, B, C, D and θ for columns (i, j) and return the rotated L."""
    p = L.shape[0]
    x, y = L[:, i], L[:, j]
    u = x ** 2 - y ** 2
    v = 2 * x * y
    A = u.sum()
    B = v.sum()
    C = (u ** 2 - v ** 2).sum()
    D = 2 * (u * v).sum()
    theta = angulo_optimo(A, B, C, D, p)

    c, s = np.cos(theta), np.sin(theta)
    Rot = np.array([[c, -s], [s, c]])
    L_nueva = L.copy()
    # [x' y'] = [x y] · Rot  ->  x' = x cosθ + y sinθ ;  y' = -x sinθ + y cosθ
    L_nueva[:, [i, j]] = L[:, [i, j]] @ Rot
    return L_nueva, Rot, (A, B, C, D, theta)


def paso7_rotacion(L_norm, variables):
    titulo("PASO 7. Rotación varimax por pares de columnas")
    p, m = L_norm.shape
    factores = [f"F{k + 1}" for k in range(m)]
    pares = list(itertools.combinations(range(m), 2))   # (F1,F2), (F1,F3), (F2,F3)
    out(f"p = {p} variables, m = {m} factores, pares: "
        + ", ".join(f"(F{i + 1},F{j + 1})" for i, j in pares))
    out(f"ITERAR = {ITERAR}  ->  " + ("iterar hasta converger" if ITERAR else "un solo ciclo"))

    L_rot = L_norm.copy()
    T = np.eye(m)              # total accumulated rotation (m x m)
    rotaciones = []            # 2x2 matrices used (for the asserts)
    n_ciclos = MAX_CICLOS if ITERAR else 1

    for ciclo in range(1, n_ciclos + 1):
        out(f"\n######## Ciclo {ciclo} ########")
        max_theta = 0.0
        for i, j in pares:
            L_rot, Rot, (A, B, C, D, theta) = rotar_par(L_rot, i, j)
            # Accumulate the rotation in T (embedding the 2x2 in an m x m identity)
            G = np.eye(m)
            G[np.ix_([i, j], [i, j])] = Rot
            T = T @ G
            rotaciones.append(Rot)
            max_theta = max(max_theta, abs(theta))

            out(f"\n>> Par (F{i + 1}, F{j + 1})")
            out(f"   A = {A:.{DEC}f}   B = {B:.{DEC}f}   C = {C:.{DEC}f}   D = {D:.{DEC}f}")
            out(f"   θ = {theta:.{DEC}f} rad = {np.degrees(theta):.{DEC}f}°")
            mostrar(f"Matriz de rotación 2x2 del par (F{i + 1},F{j + 1})", Rot,
                    [f"F{i + 1}", f"F{j + 1}"], [f"F{i + 1}'", f"F{j + 1}'"])
            mostrar(f"L_norm después de rotar (F{i + 1},F{j + 1})", L_rot, variables, factores)
        if ITERAR and max_theta < TOL:
            out(f"\nConvergió en el ciclo {ciclo} (max |θ| = {max_theta:.2e}).")
            break

    mostrar("Matriz de rotación acumulada T (L_rot_norm = L_norm · T)", T, factores, factores)
    return L_rot, T, rotaciones


# ---------------------------------------------------------------------------
# Step 8. Clean loadings L*
# ---------------------------------------------------------------------------
def paso8_desnormalizar(L_rot_norm, h, h2_original, variables):
    titulo("PASO 8. Loadings limpios L*[i,:] = L_rot_norm[i,:] · h_i")
    L_star = L_rot_norm * h[:, None]
    m = L_star.shape[1]
    h2_star = (L_star ** 2).sum(axis=1)
    mostrar("Loadings rotados L*", L_star, variables, [f"F{k + 1}" for k in range(m)])
    mostrar("Comparación de comunalidades", np.vstack([h2_original, h2_star]),
            ["h_i² de L", "h_i² de L*"], variables)
    out(f"¿Comunalidades de L* = comunalidades de L?  {np.allclose(h2_original, h2_star)}")
    return L_star


# ---------------------------------------------------------------------------
# Step 9. Reproduced theoretical matrix
# ---------------------------------------------------------------------------
def paso9_reproducida(L_star, S, variables):
    titulo("PASO 9. Matriz teórica reproducida Σ* = L* L*^T")
    Sigma_star = L_star @ L_star.T
    mostrar("Σ* = L* L*^T (escala estandarizada / correlación)", Sigma_star, variables, variables)
    D_S = np.diag(S)
    Sigma_cov = D_S @ Sigma_star @ D_S
    mostrar("Σ_cov = D_S Σ* D_S (covarianza teórica en escala original)",
            Sigma_cov, variables, variables)
    return Sigma_star, Sigma_cov


# ---------------------------------------------------------------------------
# Step 10. Error matrix
# ---------------------------------------------------------------------------
def paso10_error(R, Sigma_star, variables):
    titulo("PASO 10. Matriz de error E = R - Σ*")
    E = R - Sigma_star
    mostrar("Matriz de error E", E, variables, variables)
    frob = np.sqrt((E ** 2).sum())
    fuera_diag = E[~np.eye(E.shape[0], dtype=bool)]
    out(f"Norma de Frobenius ||E||_F = {frob:.{DEC}f}")
    out(f"Máximo |error| fuera de la diagonal = {np.abs(fuera_diag).max():.{DEC}f}")
    out("(La diagonal de E es la unicidad 1 - h_i² de cada variable.)")
    return E


# ---------------------------------------------------------------------------
# Step 11. Factor scores (Thompson / regression)
# ---------------------------------------------------------------------------
def nombrar_factores(L_star, variables, umbral=0.5):
    """Assign each factor the area (Alg/Log/Mod) of the variables that load strongly on it."""
    areas = {"Alg": "Álgebra", "Log": "Lógica", "Mod": "Modelado"}
    nombres = []
    for k in range(L_star.shape[1]):
        fuertes = [v for v, c in zip(variables, L_star[:, k]) if abs(c) >= umbral]
        # the name combines every area whose variables load strongly on the factor
        area = " + ".join(nom for clave, nom in areas.items()
                          if any(clave in v for v in fuertes)) or "?"
        nombres.append((area, fuertes))
    return nombres


def paso11_puntajes(Z, R, L_star, variables, estudiantes):
    titulo("PASO 11. Puntajes factoriales (Thompson/regresión) F = Z R^(-1) L*")
    F = Z @ np.linalg.inv(R) @ L_star
    m = L_star.shape[1]
    mostrar(f"Puntajes factoriales F (10x{m})", F, estudiantes, [f"F{k + 1}" for k in range(m)])

    # Interpretation of the clean factors: after rotation, each pair of variables from
    # the same subject (Alg1/Alg2, Log1/Log2, Mod1/Mod2) loads high on one factor and low
    # on the others. With this data Kaiser gives m = 2 (λ3 ≈ 0.03) because Algebra and
    # Modeling correlate ~0.98: they form ONE "quantitative" factor (Algebra + Modeling)
    # and Logic is a separate factor. Names are decided in code because the order and
    # number of rotated factors are not guaranteed.
    out("\nInterpretación de los factores rotados (|carga| >= 0.5):")
    for k, (area, fuertes) in enumerate(nombrar_factores(L_star, variables)):
        out(f"  F{k + 1} = {area:9s} <- variables: {', '.join(fuertes)}")
    return F


# ---------------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------------
def verificaciones(R, V, Sigma_star, rotaciones, T):
    titulo("VERIFICACIONES (asserts)")
    assert np.allclose(np.diag(R), 1.0), "La diagonal de R no es 1"
    out("[OK] diag(R) = 1")
    assert np.allclose(V.T @ V, np.eye(V.shape[1])), "V no es ortonormal"
    out("[OK] V^T V = I (V ortonormal)")
    assert np.allclose(Sigma_star, Sigma_star.T), "Σ* no es simétrica"
    out("[OK] Σ* = L* L*^T es simétrica")
    for k, Rot in enumerate(rotaciones, 1):
        assert np.isclose(np.linalg.det(Rot), 1.0), f"Rotación {k}: det != 1"
        assert np.allclose(Rot.T @ Rot, np.eye(2)), f"Rotación {k}: no ortogonal"
    out(f"[OK] Las {len(rotaciones)} rotaciones 2x2 son ortogonales con det = 1")
    assert np.allclose(T.T @ T, np.eye(T.shape[0])) and np.isclose(np.linalg.det(T), 1.0)
    out(f"[OK] Rotación acumulada T ortogonal, det(T) = {np.linalg.det(T):.{DEC}f}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    df = cargar_datos()
    variables = list(df.columns)
    estudiantes = list(df.index)
    X = df.to_numpy(dtype=float)

    titulo("DATOS ORIGINALES X (10 estudiantes x 6 variables)")
    mostrar("Matriz de datos X", X, estudiantes, variables)

    Z, media, S = paso1_estandarizar(X, variables, estudiantes)
    R = paso2_correlacion(Z, variables)
    lam, V, m = paso3_espectral(R, variables)
    L, h2 = paso4_loadings(lam, V, m, variables)
    paso5_factor_g(L, variables)
    L_norm, h = paso6_normalizar(L, h2, variables)
    L_rot_norm, T, rotaciones = paso7_rotacion(L_norm, variables)
    L_star = paso8_desnormalizar(L_rot_norm, h, h2, variables)
    Sigma_star, Sigma_cov = paso9_reproducida(L_star, S, variables)
    paso10_error(R, Sigma_star, variables)
    paso11_puntajes(Z, R, L_star, variables, estudiantes)
    verificaciones(R, V, Sigma_star, rotaciones, T)

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINEAS) + "\n")
    print(f"\nSalida guardada en {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
