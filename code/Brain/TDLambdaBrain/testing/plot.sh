import sys

def main():
  if len(sys.argv) == 1:
    print("csv file required")
    sys.exit()
  import plotext.plot as plt
  import pandas as pd
  for filename in sys.argv[1:]:
    data = pd.read_csv(filename,header=None)
    column = data.iloc[:,0].values
    plt.plot(column)
    #plt.set_equations(False)
    plt.show()

if __name__ == "__main__":
  main()
