import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

def scarica_dati(ticker, start_date, end_date=None, interval='1d'):
    """Scarica i dati da Yahoo Finance. Usa la data di oggi come end_date di default."""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')  # Usa la data di oggi se end_date non è specificata
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"Errore: nessun dato trovato per il ticker {ticker}. Verifica il ticker o le date inserite.")
            return None
        return data
    except Exception as e:
        print(f"Errore nel download dei dati: {e}")
        return None

def calcola_var(data, confidence_level=0.99):
    """Calcola il VaR utilizzando 3 metodi: Parametrico, Monte Carlo e Storico."""
    mean_return = data['Return'].mean()
    std_dev = data['Return'].std()
    
    # 1. Metodo Parametrico
    z_score = norm.ppf(1 - confidence_level)
    var_parametrico = np.abs(z_score * std_dev)
    
    # 2. Metodo della Simulazione Monte Carlo
    MC = 10000
    simulazioni = np.random.normal(mean_return, std_dev, MC)
    var_montecarlo = -np.percentile(simulazioni, (1 - confidence_level) * 100)
    
    # 3. Metodo Storico
    var_storico = -np.percentile(data['Return'].dropna(), (1 - confidence_level) * 100)
    
    return var_parametrico, var_montecarlo, var_storico

# Esegui l'analisi
ticker_symbol = input("Inserisci il ticker (es. 'AAPL', 'OXY'): ")
start_date = input("Inserisci la data di inizio (es. '2000-01-01'): ")
data = scarica_dati(ticker_symbol, start_date)

if data is not None:
    # Calcolo Log Returns
    data['Return'] = np.append([np.nan], np.diff(np.log(data['Adj Close'])))
    data.dropna(inplace=True)
    
    # Statistiche descrittive
    print("Statistiche descrittive dei rendimenti:")
    print(data['Return'].describe())

    # Calcola VaR
    var_parametrico, var_montecarlo, var_storico = calcola_var(data)

    print(f"VaR al 99% (Metodo Parametrico): {var_parametrico}")
    print(f"VaR al 99% (Simulazione Monte Carlo): {var_montecarlo}")
    print(f"VaR al 99% (Metodo Storico): {var_storico}")

    # Visualizzazione dei grafici
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(data.index, data['Adj Close'], color='green')
    plt.title(f"Prezzo Adjusted di '{ticker_symbol}'")
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.grid()
    plt.show()

    # Plot Distribuzione dei Rendimenti
    plt.figure(figsize=(10, 5), dpi=100)
    plt.hist(data['Return'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Distribuzione dei Rendimenti')
    plt.xlabel('Frequenza')
    plt.ylabel('Return')
    plt.grid()
    plt.axvline(data['Return'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean Return: {data['Return'].mean():.4f}")
    plt.legend(loc='upper right')
    plt.show()

    # Plot Distribuzione dei Rendimenti con VaR
    plt.figure(figsize=(10, 5), dpi=100)
    plt.hist(data['Return'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribuzione dei Rendimenti con VaR al 99%')
    plt.xlabel('Return')
    plt.ylabel('Frequenza')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(-var_parametrico, color='red', linestyle='--', linewidth=1.5, label=f"VaR Parametrico: {-var_parametrico:.4f}")
    plt.axvline(-var_montecarlo, color='green', linestyle='--', linewidth=1.5, label=f"VaR Monte Carlo: {-var_montecarlo:.4f}")
    plt.axvline(-var_storico, color='purple', linestyle='--', linewidth=1.5, label=f"VaR Storico: {-var_storico:.4f}")
    plt.legend(loc='upper right')
    plt.show()


