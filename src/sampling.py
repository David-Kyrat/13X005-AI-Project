from naive_bayes import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from metrics import compute_metrics
import softmax
from matplotlib import pyplot as plt
from main import FEAT, LABELS
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# ECHANTILLONNAGE

# Prendre 50 échantillons de chaque classe
SAMPLES = 50
parameters = get_distrib_parameters(FEAT, LABELS)
classes = parameters.keys()

sampled_data = []
sampled_labels = []

# On génère des échantillons pour chaque classe
for y in classes:
    class_samples = []

    for variable_params in parameters[y]:
        mean, std = variable_params
        samples = np.random.normal(mean, std, SAMPLES)
        class_samples.append(samples)

    class_samples = np.column_stack(class_samples)
    sampled_data.append(class_samples)
    sampled_labels.extend([y] * SAMPLES)

# On concatène les échantillons
sampled_data = np.vstack(sampled_data)
sampled_labels = np.array(sampled_labels)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# COMPARAISON

for c in classes:
    means, stds = zip(*parameters[c])
    print(f"Classe {c} réelle")
    print(f"Mean: {means}")
    print(f"Std: {stds}")

    mean_sampled = np.mean(sampled_data[sampled_labels == c], axis=0)
    std_sampled = np.std(sampled_data[sampled_labels == c], axis=0)

    print(f"Classe {c} échantillonnée")
    print(f"Mean: {mean_sampled}")
    print(f"Std: {std_sampled}")

    vars = [0, 1, 2, 3]

    plt.figure()
    plt.title(f"Courbe des distribution de probabilité sachant Y={c}")

    for var in vars:
        plt.plot(
            np.linspace(means[var] - 10, means[var] + 10, 1000),
            normal_pdf(means[var], stds[var])(np.linspace(means[var] - 10, means[var] + 10, 1000)),
            label=f"X_{var} réelle",
        )
        plt.plot(
            np.linspace(mean_sampled[var] - 10, mean_sampled[var] + 10, 1000),
            normal_pdf(mean_sampled[var], std_sampled[var])(
                np.linspace(mean_sampled[var] - 10, mean_sampled[var] + 10, 1000)
            ),
            label=f"X_{var} échantillonnée",
        )

    plt.legend()
    plt.savefig(f"src/res/sample_compare_Y_{c}")
    plt.close()


print("\n\n")

X_train, X_test, y_train, y_test = train_test_split(sampled_data, sampled_labels, test_size=0.3, random_state=42)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

sampled_params = {
    c: list(zip(np.mean(X_train[y_train == c], axis=0), np.std(X_train[y_train == c], axis=0))) for c in classes
}

# --- Notre implémentation de Naive Bayes ---
print("Notre Naive Bayes")
predicted_nb = predict_bayes_all(X_test, sampled_params)
print(compute_metrics(y_test, predicted_nb))

# --------------------------------------------

# --- SKLearn Naive Bayes ---
print("Sklearn Naive Bayes")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predicted_gnb = gnb.predict(X_test)
print(compute_metrics(y_test, predicted_gnb))

# ----------------------------

# --- Notre implémentation de Logistic Regression ---
print("Notre Logistic Regression")
theta = softmax.train_log_reg_2(X_train, y_train, np.zeros((len(classes), X_train.shape[1] + 1)), 1000, 1e-4)
predicted_logreg = softmax.predict_log_reg_2(X_test, theta)
print(compute_metrics(y_test, predicted_logreg))


# --------------------------------------------

# --- SKLearn Logistic Regression ---
print("Sklearn Logistic Regression")
lr = LogisticRegression(multi_class="multinomial")
lr.fit(X_train, y_train)
predicted_lr = lr.predict(X_test)
print(compute_metrics(y_test, predicted_lr))

# --------------------------------------------

# ---------------------------------------------------------------------------
