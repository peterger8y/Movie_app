{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of Untitled6.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "nbI6UhHFJQtA"
      },
      "source": [
        "#importing libraries, installing category encoders:\n",
        "#also loading relevant data frames\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "\n",
        "df = pd.read_csv('/content/movies_metadata.csv', low_memory=False)\n",
        "df.loc[df['release_date'] == 'TV Movie 2019', 'release_date'] = '2019'\n",
        "df['release_date'] = pd.to_datetime(df['release_date'], infer_datetime_format=True, errors='coerce')\n",
        "df.drop(columns='adult', inplace=True)\n",
        "df['year'] = df['release_date'].dt.year\n",
        "df['year'] = df['year'].astype(float)\n",
        "df_fresh = pd.read_csv('/content/IMDb movies.csv', low_memory=False)\n",
        "df_fresh.loc[df_fresh['year'] == 'TV Movie 2019', 'year'] = 2019\n",
        "df_fresh['year'] = df_fresh['year'].astype(float)\n",
        "a = df.loc[df['revenue'].isnull()].index\n",
        "df.drop(labels=a, inplace=True)\n",
        "df.reset_index(inplace=True)\n",
        "df.drop_duplicates(inplace=True)"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mRyjW6ygiohz",
        "outputId": "f2893230-8bd9-4d24-fc72-b09404501164"
      },
      "source": [
        "!pip install category_encoders"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: category_encoders in /usr/local/lib/python3.7/dist-packages (2.2.2)\n",
            "Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.22.2.post1)\n",
            "Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.19.5)\n",
            "Requirement already satisfied: pandas>=0.21.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.1.5)\n",
            "Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.10.2)\n",
            "Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.5.1)\n",
            "Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.4.1)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.0.1)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2.8.1)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2018.9)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from patsy>=0.5.1->category_encoders) (1.15.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uFtnSFlhiGZ8"
      },
      "source": [
        "#Decreasing number of NAN inputs for revenue referencing another IMDB dataset\n",
        "\n",
        "alpha = df.loc[df['revenue'] == 0, 'title']\n",
        "\n",
        "for x in alpha:\n",
        "  b = df_fresh.loc[df_fresh['title'] == x, 'worlwide_gross_income']\n",
        "  c = df_fresh.loc[df_fresh['title'] == x, 'year']\n",
        "  for y, z in enumerate(b):\n",
        "    df.loc[(df['original_title']==x) & (df['year'] ==c.iloc[y]), 'revenue'] = b.iloc[y]\n",
        "\n",
        "alpha = df.loc[df['revenue'] == 0].index\n",
        "df.drop(labels=alpha, inplace=True)"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s_rczYdIo14S"
      },
      "source": [
        "#manufacturing new columns for previous dataset to improve machine learning model\n",
        "#Again, pulling from the same IMDB dataset:\n",
        "\n",
        "df_act_dir = pd.read_csv('/content/IMDb movies.csv', low_memory=False)\n",
        "\n",
        "df_act_dir.drop(columns = ['imdb_title_id', 'date_published', 'genre', 'country', 'language', \n",
        "                           'description', 'avg_vote', 'votes', 'budget', 'usa_gross_income', 'worlwide_gross_income', \n",
        "                           'metascore', 'reviews_from_users', 'reviews_from_critics', 'original_title'], inplace=True)\n",
        "df = pd.merge(df, df_act_dir, how='inner', on='title')"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KsD8k44NJQtD"
      },
      "source": [
        "#dropping high cardinality features, and features presently unusable\n",
        "\n",
        "features_to_drop = ['belongs_to_collection', 'homepage', 'imdb_id', 'overview', \n",
        "                    'tagline', 'title', 'poster_path', 'original_title', 'production_companies', 'status']\n",
        "df_x = df.drop(columns = features_to_drop)"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o6vmaVS9HN7J"
      },
      "source": [
        "#dropping null revenue rows:\n",
        "a = df_x.loc[df['revenue'].isnull()].index\n",
        "df_x.drop(labels=a, inplace=True)\n",
        "df_x.drop(columns=['year_x', 'year_y'], inplace=True)"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lwFQx7DfJQtE"
      },
      "source": [
        "# cleaning up production_countries and genres columns, making them more readable\n",
        "ak = []\n",
        "for x in df['production_countries']:\n",
        "    x = x.split(':')\n",
        "    if len(x)<=1:\n",
        "        ak.append('unknown')\n",
        "    else:\n",
        "        x = x[-1].strip('}]')\n",
        "        x = x.strip()\n",
        "        ak.append(x)\n",
        "\n",
        "df['production_countries'] = ak\n",
        "\n",
        "ab = []\n",
        "ad = []\n",
        "for x in range(len(df['genres'])):\n",
        "    ac = []\n",
        "    ae = []\n",
        "    y = df['genres'][x]\n",
        "    g = y.split(':')\n",
        "    for z in range(len(g)):\n",
        "        if z%2 == 1:\n",
        "            h = g[z].split()\n",
        "            h = h[0].strip(',')\n",
        "            ac.append(h)\n",
        "    ab.append(ac)\n",
        "\n",
        "            \n",
        "df['genres'] = ab \n",
        "\n",
        "ab = []\n",
        "for x in df['genres']:\n",
        "    if len(x) == 0:\n",
        "        ab.append(0)\n",
        "    else:\n",
        "        for y in range(len(x)):\n",
        "            if y ==0:\n",
        "                g = x[y]\n",
        "            else:\n",
        "                g = g + x[y]\n",
        "        ab.append(g)\n",
        "\n",
        "df['genres'] = ab  "
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KCbnw5X7GHHf"
      },
      "source": [
        "#cleaning up revenue column for machine learning model:\n",
        "\n",
        "ab = []\n",
        "df_x['revenue'] = df_x['revenue'].astype(str)\n",
        "for x in df_x['revenue']:\n",
        "  x = x.strip('$')\n",
        "  x = x.strip('INR')\n",
        "  ab.append(x)\n",
        "\n",
        "df_x['revenue'] = ab\n",
        "df_x['revenue'] = df_x['revenue'].astype(float)\n",
        "\n",
        "#Generating new columns for Year, month, and day\n",
        "\n",
        "df_x['release_date'] = pd.to_datetime(df_x['release_date'], infer_datetime_format=True)\n",
        "df_x['year'] = df_x['release_date'].dt.year\n",
        "df_x['month'] = df_x['release_date'].dt.month\n",
        "df_x['day'] = df_x['release_date'].dt.day\n",
        "df_x.drop(columns='release_date', inplace=True)"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wmeY0erHoO6h"
      },
      "source": [
        "#importing another IMDB dataset to reference for Actor and actresses column in present dataset:\n",
        "#also, converting actor column from difficult to interpret strings into a measure of how many \n",
        "#'top' actresses and actors are present in film\n",
        "\n",
        "top_1000 = pd.read_csv('/content/Top 1000 Actors and Actresses.csv', encoding='latin-1')\n",
        "\n",
        "abb = []\n",
        "for x in top_1000['Name']:\n",
        "  abb.append(x)\n",
        "\n",
        "df_x.loc[df_x['actors'].isnull(), 'actors'] = 'none'\n",
        "\n",
        "ab = []\n",
        "\n",
        "for x in df_x['actors']:\n",
        "  ac = []\n",
        "  x = x.split(',')\n",
        "  for y in x:\n",
        "    y = y.strip()\n",
        "    if y in abb:\n",
        "      ac.append(1)\n",
        "  b = np.sum(ac)\n",
        "  ab.append(b)\n",
        "\n",
        "df_x['actors'] = ab"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VGoaGd_5M2Vi"
      },
      "source": [
        "#reducing cardinality of production_company feature:\n",
        "a = pd.DataFrame(df_x['production_company'].value_counts())\n",
        "for x in a.index:\n",
        "  if a.loc[x][0]< 30:\n",
        "    df_x.loc[df_x['production_company'] == x, 'production_company'] = 'other'"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jkLqE-yJ-3WS"
      },
      "source": [
        "#performing same action with directors as with actors:\n",
        "top_dir = pd.read_csv('/content/1000 the Best Directors.csv', encoding='latin-1')\n",
        "abb = []\n",
        "for x in top_dir['Name']:\n",
        "  abb.append(x)\n",
        "\n",
        "df_x['director'] = df_x['director'].astype(str)\n",
        "ab = []\n",
        "for x in df_x['director']:\n",
        "  ac = []\n",
        "  x = x.split(',')\n",
        "  for y in x:\n",
        "    y = y.strip()\n",
        "    if y in abb:\n",
        "      ac.append(1)\n",
        "  b = np.sum(ac)\n",
        "  ab.append(b)\n",
        "\n",
        "df_x['director'] = ab"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xBZ1_DkwA1uR"
      },
      "source": [
        "#dropping low importance feature....\n",
        "\n",
        "df_x = df_x.drop(columns='writer')"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TxTJZDSG9wTG"
      },
      "source": [
        "#cleaning up spoken_languages feature for easier interpetability\n",
        "\n",
        "\n",
        "ab = []\n",
        "for x in df_x['spoken_languages']:\n",
        "  x = x.split(':')\n",
        "  y = len(x)\n",
        "  z = y//2\n",
        "  ab.append(z)\n",
        "\n",
        "df_x['number_of_languages'] = ab"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GPBPxF-MOnP4"
      },
      "source": [
        "#cleaning up genres\n",
        "\n",
        "ac = []\n",
        "for x in df_x['genres']:\n",
        "  ab = []\n",
        "  x = x.split(':')\n",
        "  for y, z in enumerate(x):\n",
        "    if y != 0:\n",
        "      if y%2 ==0:\n",
        "        z = z.split(',')\n",
        "        ab.append(z[0])\n",
        "  ac.append(ab)"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jEwAkONCS0eR"
      },
      "source": [
        "#cleaning up genres:\n",
        "\n",
        "ab = []\n",
        "for x in ac:\n",
        "  ad = []\n",
        "  if len(x) == 0:\n",
        "    ab.append('uknown')\n",
        "  else:\n",
        "    for y in x:\n",
        "      y = y.strip('\"')\n",
        "      y = y.strip('}')\n",
        "      ad.append(y)\n",
        "    ab.append(ad)\n"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8JDQ2E4iT8Yx"
      },
      "source": [
        "#cleaning up genres, cont.:\n",
        "ac = []\n",
        "for x in ab:\n",
        "  ad = []\n",
        "  for y in x:\n",
        "    y = y.strip('}]')\n",
        "    y = y.strip('\"')\n",
        "    ad.append(y)\n",
        "  ac.append(ad)"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zLFSBfquVNYz"
      },
      "source": [
        "#cleaning up genres, cont.:\n",
        "ab = []\n",
        "for x in ac:\n",
        "  ad = []\n",
        "  for y in x:\n",
        "    y = y.strip(\"'\")\n",
        "    y = y.strip(\"' \")\n",
        "    ad.append(y)\n",
        "  ab.append(ad)"
      ],
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EJhr2rJ-VqGq"
      },
      "source": [
        "#cleaning u genres, cont.:\n",
        "ac = []\n",
        "for x in ab:\n",
        "  if len(x) == 0:\n",
        "    ac.append('unknown')\n",
        "  else:\n",
        "    for y, z in enumerate(x):\n",
        "      if y == 0:\n",
        "        w = z\n",
        "      else:\n",
        "        w = w + (', ') + z\n",
        "    ac.append(w)"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pRs19xMYWbrj"
      },
      "source": [
        "#cleaned genres:\n",
        "df_x['genres'] = ac"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SLD8O30W-4Hn"
      },
      "source": [
        "#since spoken languages was converted into number of languages released, dropped lesser\n",
        "#important feature:\n",
        "df_x.drop(columns='spoken_languages', inplace=True)"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ov1dowTsWk7o"
      },
      "source": [
        "#dropping junk features:\n",
        "df_x.drop(columns=['index', 'video', 'id'], inplace=True)"
      ],
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ozi-n0SeZMuI"
      },
      "source": [
        "#cleaning production countries feature:\n",
        "ab = []\n",
        "for x in df_x['production_countries']:\n",
        "  x = x.split(':')\n",
        "  x = x[-1]\n",
        "  x = x.strip(\"' \")\n",
        "  x = x.strip(\"}]'\")\n",
        "  ab.append(x)"
      ],
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1ymSdJdf98GG"
      },
      "source": [
        "#cleaning genre feature:\n",
        "ab = []\n",
        "for x in df_x['genres']:\n",
        "  if x == 'u, k, o, w, n':\n",
        "    ab.append('unknown')\n",
        "  else:\n",
        "    x = x.split(',')\n",
        "    ab.append(x[0])\n",
        "df_x['genres'] = ab"
      ],
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zSiKmbI4Zx31"
      },
      "source": [
        "df_x['production_countries'] = ab"
      ],
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TSCqAMP_Z2wx"
      },
      "source": [
        "df_x.loc[df_x['production_countries'] == '[', 'production_countries'] = 'unknown'"
      ],
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IA5uWW3hJKX1"
      },
      "source": [
        "#prepping budget feature\n",
        "df_x['budget'] = df_x['budget'].astype(float)\n",
        "mean = df_x['budget'].loc[df_x['budget'] != 0].mean()\n",
        "df_x.loc[df_x['budget']==0, 'budget'] = mean"
      ],
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 341
        },
        "id": "3VlhpcNUafWY",
        "outputId": "01a06d74-aa04-4dd2-a2da-4bba0d407549"
      },
      "source": [
        "# removing redundant column:\n",
        "df_x.drop(columns='duration', inplace=True)"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "error",
          "ename": "KeyError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-53-2f2f0a179dfe>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# removing redundant column:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mdf_x\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'duration'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, axis, index, columns, level, inplace, errors)\u001b[0m\n\u001b[1;32m   4172\u001b[0m             \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4173\u001b[0m             \u001b[0minplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minplace\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4174\u001b[0;31m             \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   4175\u001b[0m         )\n\u001b[1;32m   4176\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, axis, index, columns, level, inplace, errors)\u001b[0m\n\u001b[1;32m   3887\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;32min\u001b[0m \u001b[0maxes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3888\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3889\u001b[0;31m                 \u001b[0mobj\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_drop_axis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3890\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3891\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0minplace\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_drop_axis\u001b[0;34m(self, labels, axis, level, errors)\u001b[0m\n\u001b[1;32m   3921\u001b[0m                 \u001b[0mnew_axis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3922\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3923\u001b[0;31m                 \u001b[0mnew_axis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3924\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreindex\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0maxis_name\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mnew_axis\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3925\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mdrop\u001b[0;34m(self, labels, errors)\u001b[0m\n\u001b[1;32m   5285\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmask\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0many\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5286\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0merrors\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m\"ignore\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 5287\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"{labels[mask]} not found in axis\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   5288\u001b[0m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mindexer\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m~\u001b[0m\u001b[0mmask\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5289\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdelete\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyError\u001b[0m: \"['duration'] not found in axis\""
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KY9alBggJfTM"
      },
      "source": [
        "#removing potential feature leakage:\n",
        "df_x.head()\n",
        "df_x.drop(columns=['popularity', 'vote_average', 'vote_count'], inplace=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zQdLpdSNqsSs"
      },
      "source": [
        "X = df_x.drop(columns='revenue')\n",
        "y = df_x['revenue']"
      ],
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BACdx-_VJQtJ"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "xtrain, X_test, ytrain, y_test = train_test_split(X, y, test_size = .2, random_state=42)\n",
        "X_train, X_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size = .2, random_state=42)"
      ],
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o7KjbO0XJQtJ"
      },
      "source": [
        "baseline = np.mean(y_train)\n",
        "baseline_train = [baseline for x in range(len(y_train))]\n",
        "baseline_train = np.array(baseline_train)\n",
        "baseline_acc = mean_absolute_error(y_train, baseline_train)"
      ],
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a8PtZVCpJQtK",
        "outputId": "a871111e-d582-4947-9bdf-b082ba24693e"
      },
      "source": [
        "baseline_acc"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "52169022.79024799"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ewCThi5HJQtK",
        "outputId": "dff8e593-e636-4df8-ca97-81b82e6c97cf"
      },
      "source": [
        "#Importing packages and fitting a Random Forest Regressor\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.metrics import r2_score\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "from category_encoders import OrdinalEncoder\n",
        "\n",
        "pipeline=make_pipeline(OrdinalEncoder(cols=['original_language', 'production_countries', 'production_company', 'genres']), SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators = 150, min_samples_leaf=5, random_state=41))\n",
        "pipeline.fit(X_train, y_train)\n",
        "\n",
        "pipeline.score(X_train, y_train)"
      ],
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.7868470618062438"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lSwkgBzRJQtL",
        "outputId": "443ef4bc-ec53-47cb-b045-6d356f522afe"
      },
      "source": [
        "pipeline.score(X_val, y_val)"
      ],
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6715675507309099"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "id": "zO5bnxdr9c80",
        "outputId": "ad213209-e224-4538-d0e3-cdfa663870c6"
      },
      "source": [
        "y_pred = pipeline.predict(X_val)\n",
        "display(mean_absolute_error(y_val, y_pred))\n",
        "display(mean_squared_error(y_val, y_pred))"
      ],
      "execution_count": 83,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "21960608.144722156"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "4335676575330818.0"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-_Qy7rYEfOTT",
        "outputId": "08dab120-7367-47fc-db47-b0057ba34e64"
      },
      "source": [
        "from joblib import dump\n",
        "dump(pipeline, 'pipline.joblib', compress = True)"
      ],
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['pipline.joblib']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Mcjmjhwsf_4a",
        "outputId": "b4d79354-f6bf-4e28-d72f-eace49f696be"
      },
      "source": [
        "#exporting trained model\n",
        "import joblib\n",
        "import sklearn\n",
        "import category_encoders as ce\n",
        "import xgboost\n",
        "print(f'joblib=={joblib.__version__}')\n",
        "print(f'scikit-learn=={sklearn.__version__}')\n",
        "print(f'category_encoders=={ce.__version__}')\n",
        "print(f'xgboost=={xgboost.__version__}')"
      ],
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "joblib==1.0.1\n",
            "scikit-learn==0.22.2.post1\n",
            "category_encoders==2.2.2\n",
            "xgboost==0.90\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q_a2FvGtFa4k"
      },
      "source": [
        "df_x.to_csv('first_exp')"
      ],
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WtWV9JRSsuFe",
        "outputId": "fcbe8284-59aa-4b70-dd85-9e121cb590fc"
      },
      "source": [
        "!pip install pdpbox"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: pdpbox in /usr/local/lib/python3.7/dist-packages (0.2.0)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from pdpbox) (0.22.2.post1)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from pdpbox) (1.19.5)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from pdpbox) (1.4.1)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from pdpbox) (1.1.5)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.7/dist-packages (from pdpbox) (5.4.8)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from pdpbox) (1.0.1)\n",
            "Requirement already satisfied: matplotlib>=2.1.2 in /usr/local/lib/python3.7/dist-packages (from pdpbox) (3.2.2)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->pdpbox) (2018.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->pdpbox) (2.8.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.1.2->pdpbox) (1.3.1)\n",
            "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.1.2->pdpbox) (2.4.7)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.1.2->pdpbox) (0.10.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->pdpbox) (1.15.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_HsqvQ8Ys_se"
      },
      "source": [
        "from pdpbox.pdp import pdp_isolate, pdp_plot\n",
        "\n",
        "feature = 'year'\n",
        "\n",
        "isolated = pdp_isolate(\n",
        "    model=pipeline, \n",
        "    dataset=X_val, \n",
        "    model_features=X_val.columns, \n",
        "    feature=feature,\n",
        "    num_grid_points=50\n",
        ")"
      ],
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yqw6ihcJuorO"
      },
      "source": [
        "#generating dataframe for pdp outputs in app\n",
        "from random import random\n",
        "from random import sample"
      ],
      "execution_count": 66,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CDMfh_zXu0K7"
      },
      "source": [
        "X_val.reset_index(inplace=True)"
      ],
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FzeTXzUSu8aw"
      },
      "source": [
        "X_val.drop(columns='index', inplace=True)"
      ],
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yVplpn1UvEyt"
      },
      "source": [
        "index = X_val.index"
      ],
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JPMJ9wSovTH0"
      },
      "source": [
        "index = list(index)"
      ],
      "execution_count": 70,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wfB3wTvXvBEm"
      },
      "source": [
        "sample_set = sample(index, k=50)"
      ],
      "execution_count": 71,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 375
        },
        "id": "TvhG2lgrvYPp",
        "outputId": "03d09256-a202-416e-dc35-534fae0f1651"
      },
      "source": [
        "for x, y in enumerate(sample_set):\n",
        "  for w, z in enumerate(X_val.columns):\n",
        "    for u, v in enumerate(X_val[z].value_counts().index):\n",
        "       if x == 0:\n",
        "        if w == 0:\n",
        "          if u == 0:\n",
        "            df_pdp = X_val.iloc[[y]]\n",
        "            df_pdp['indicator'] = z\n",
        "            df_pdp[z] = v\n",
        "       else:\n",
        "          df_temp = X_val.iloc[[y]]\n",
        "          df_temp['indicator'] = z\n",
        "          df_temp[z] = v\n",
        "          df_pdp = pd.concat([df_pdp, df_temp], ignore_index=True)"
      ],
      "execution_count": 72,
      "outputs": [
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-72-0114cda09be3>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     11\u001b[0m           \u001b[0mdf_temp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX_val\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m           \u001b[0mdf_temp\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'indicator'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mz\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m           \u001b[0mdf_temp\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mz\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mv\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m           \u001b[0mdf_pdp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mdf_pdp\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdf_temp\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__setitem__\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   3042\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3043\u001b[0m             \u001b[0;31m# set column\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3044\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_set_item\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3045\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3046\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_setitem_slice\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mslice\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m_set_item\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   3119\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ensure_valid_index\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3120\u001b[0m         \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sanitize_column\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3121\u001b[0;31m         \u001b[0mNDFrame\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_set_item\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3122\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3123\u001b[0m         \u001b[0;31m# check if we are modifying a copy\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_set_item\u001b[0;34m(self, key, value)\u001b[0m\n\u001b[1;32m   3580\u001b[0m             \u001b[0;32mreturn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3581\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3582\u001b[0;31m         \u001b[0mNDFrame\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iset_item\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3583\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3584\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_set_is_copy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mref\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mbool_t\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m_iset_item\u001b[0;34m(self, loc, value)\u001b[0m\n\u001b[1;32m   3569\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3570\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_iset_item\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloc\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mint\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3571\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_mgr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3572\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_clear_item_cache\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3573\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/internals/managers.py\u001b[0m in \u001b[0;36miset\u001b[0;34m(self, loc, value)\u001b[0m\n\u001b[1;32m   1081\u001b[0m                     \u001b[0;32mreturn\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mplacement\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1082\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1083\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1084\u001b[0m                 raise AssertionError(\n\u001b[1;32m   1085\u001b[0m                     \u001b[0;34m\"Shape of new values must be compatible with manager shape\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/internals/managers.py\u001b[0m in \u001b[0;36mshape\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    212\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    213\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mTuple\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mint\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m...\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 214\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mtuple\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0max\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maxes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    215\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    216\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/core/internals/managers.py\u001b[0m in \u001b[0;36m<genexpr>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m    212\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    213\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mTuple\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mint\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m...\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 214\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mtuple\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0max\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maxes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    215\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    216\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LwhCwGqY1H_V"
      },
      "source": [
        "pdp_prep = df_pdp.drop(columns='indicator')\n",
        "pdp_pred = pipeline.predict(pdp_prep)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mhhOioxh1aAl"
      },
      "source": [
        "df_pdp['pred'] = pdp_pred"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NSrP70nV1k7A"
      },
      "source": [
        "df_pdp.columns"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1jJHIS3jyyZu"
      },
      "source": [
        "import seaborn as sns"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o4BnOvHx04xN"
      },
      "source": [
        "sns.lineplot(x='actors', y='pred', data=df_pdp.loc[df_pdp['indicator'] == 'actors'])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-l-N6Dg-2aBo"
      },
      "source": [
        "for x, y in enumerate(df_pdp.columns):\n",
        "  if x == 0:\n",
        "    df_temp = df_pdp.loc[df_pdp['indicator'] == y]\n",
        "    df_app_pdp = df_temp.groupby(y).mean()\n",
        "    df_app_pdp[y] = df_app_pdp.index\n",
        "    df_app_pdp['indicator'] = y\n",
        "  else:\n",
        "    df_temp = df_pdp.loc[df_pdp['indicator'] == y]\n",
        "    df_temp = df_temp.groupby(y).mean()\n",
        "    df_temp['indicator'] = y\n",
        "    df_temp[y] = df_temp.index\n",
        "    df_app_pdp = pd.concat([df_app_pdp, df_temp], ignore_index=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AqpKXPr84fmU"
      },
      "source": [
        "sns.lineplot(x = df_app_pdp.loc[df_app_pdp['indicator'] == 'actors', 'actors'], y = df_app_pdp.loc[df_app_pdp['indicator'] == 'actors', 'pred'], data=df_app_pdp)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GRMAzzlg7859"
      },
      "source": [
        "df_app_pdp.loc[df_app_pdp['indicator'] == 'actors']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qAv8Wte8-5on"
      },
      "source": [
        "df_app_pdp.to_csv('df_app_pdp.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7tyuO_J4lrGG"
      },
      "source": [
        "df_pdp.to_csv('hello.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0yQj5eLbl7bp"
      },
      "source": [
        "df_x.to_csv('howdy.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4FUKtxKk9pH_",
        "outputId": "57340303-457c-4f50-e00d-d24858670b4e"
      },
      "source": [
        "#final test to evaluate model\n",
        "pipeline.score(X_test, y_test)"
      ],
      "execution_count": 77,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6026499817997271"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 77
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "k7e8cxTF-o85"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}