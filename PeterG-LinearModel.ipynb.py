{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled24.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "axEamTe88loQ"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.linear_model import Ridge\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.metrics import r2_score\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "from sklearn.pipeline import Pipeline"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7FF04_vUwbCm"
      },
      "source": [
        "df = pd.read_csv('/content/howdy.csv')\n",
        "df.drop(columns='Unnamed: 0', inplace=True)"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DbzkALWYxdy5"
      },
      "source": [
        "ab = []\n",
        "for x in df['genres']:\n",
        "  if x == 'u, k, o, w, n':\n",
        "    ab.append('unknown')\n",
        "  else:\n",
        "    x = x.split(',')\n",
        "    ab.append(x[0])\n",
        "df['genres'] = ab"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kP2gBTid4IUP"
      },
      "source": [
        "ab = []\n",
        "for x in df['original_language']:\n",
        "  if x == 'en':\n",
        "    ab.append(1)\n",
        "  else:\n",
        "    ab.append(0)\n",
        "df['original_language'] = ab"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pdXs_5LK1I-i"
      },
      "source": [
        "df.loc[df['genres'] == 'u', 'genres'] = 'unknown'"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gV8H8sZZrfYe",
        "outputId": "c1ec2e39-48ce-4520-b0dd-bdc2a2869a4f"
      },
      "source": [
        "!pip install category_encoders"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting category_encoders\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/44/57/fcef41c248701ee62e8325026b90c432adea35555cbc870aff9cfba23727/category_encoders-2.2.2-py2.py3-none-any.whl (80kB)\n",
            "\r\u001b[K     |████                            | 10kB 15.0MB/s eta 0:00:01\r\u001b[K     |████████▏                       | 20kB 20.4MB/s eta 0:00:01\r\u001b[K     |████████████▏                   | 30kB 16.5MB/s eta 0:00:01\r\u001b[K     |████████████████▎               | 40kB 9.6MB/s eta 0:00:01\r\u001b[K     |████████████████████▎           | 51kB 8.9MB/s eta 0:00:01\r\u001b[K     |████████████████████████▍       | 61kB 8.1MB/s eta 0:00:01\r\u001b[K     |████████████████████████████▍   | 71kB 8.4MB/s eta 0:00:01\r\u001b[K     |████████████████████████████████| 81kB 4.6MB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.19.5)\n",
            "Requirement already satisfied: pandas>=0.21.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.1.5)\n",
            "Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.22.2.post1)\n",
            "Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.5.1)\n",
            "Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.4.1)\n",
            "Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.10.2)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2.8.1)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2018.9)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.0.1)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from patsy>=0.5.1->category_encoders) (1.15.0)\n",
            "Installing collected packages: category-encoders\n",
            "Successfully installed category-encoders-2.2.2\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2PlaV0Z_8nml",
        "outputId": "f8c8eabb-95a9-429d-9646-dbb43fb574f5"
      },
      "source": [
        "from category_encoders import OrdinalEncoder\n",
        "from category_encoders import OneHotEncoder\n",
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.\n",
            "  import pandas.util.testing as tm\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Dt6aP8CL2dhA"
      },
      "source": [
        "#df.drop(columns='original_language', inplace=True)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zVFCgP1asiGK"
      },
      "source": [
        "X = df.drop(columns='revenue')\n",
        "y = df['revenue']"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9E9SptwotCt2"
      },
      "source": [
        "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1)"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eVXusS0dwDfJ"
      },
      "source": [
        "from sklearn.preprocessing import PolynomialFeatures"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c7kDnnIar-mS"
      },
      "source": [
        "model = make_pipeline(OneHotEncoder(), SimpleImputer(), StandardScaler(), Ridge(alpha = 500))\n"
      ],
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0Xzm5tRssLlz",
        "outputId": "4a76d879-6b79-46ea-f672-b3efd2c748a2"
      },
      "source": [
        "model.fit(X_train, y_train)\n",
        "model.score(X_val, y_val)"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/category_encoders/utils.py:21: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead\n",
            "  elif pd.api.types.is_categorical(cols):\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.516149630511689"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X2iHqaTDviUj",
        "outputId": "76111b69-0094-4230-f17f-246bd02667ef"
      },
      "source": [
        "model.score(X_train, y_train)"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.5305230528453723"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mjjahSmPvpdp"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}