#========================IMPORTS=========================
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.optimize import linprog
import pandas as pd
import sys

#========================HELPER FUNCTIONS=========================

# returns True if the date is the end of the month 
def is_end_of_month(d: date) -> bool:
    return (d + timedelta(days=1)).month != d.month

# returns a list of coupon dates from settlementDate to maturityDate
def getCouponDates(settlementDate: date, maturityDate: date) -> list[date]:
    coupon_dates = [maturityDate]
    is_eom_bond = is_end_of_month(maturityDate)
    current_date = maturityDate

    while True:
        current_date -= relativedelta(months=6)

        if is_eom_bond:
            next_month = current_date + relativedelta(months=1, day=1)
            current_date = next_month - timedelta(days=1)

        if current_date <= settlementDate:
            break

        coupon_dates.append(current_date)

    coupon_dates.reverse()
    return coupon_dates

# returns the dirty price
def dirtyPrice(settlementDate, maturityDate, couponRate, cleanPrice, isbill=False):

    if settlementDate >= maturityDate:
        # For this problem, we filter out bonds that mature on or before settlement,
        # but this check remains for robustness.
        return cleanPrice

    if isbill:
        return cleanPrice

    coupon_dates = getCouponDates(settlementDate, maturityDate)
    if not coupon_dates:
        # This can happen if the bond matures before the first coupon after settlement
        return cleanPrice

    next_coupon_date = coupon_dates[0]
    last_coupon_date = next_coupon_date - relativedelta(months=6)
    
    # Handle cases where settlement is before the first coupon period begins
    if settlementDate < last_coupon_date:
        return cleanPrice

    days_since_last_coupon = (settlementDate - last_coupon_date).days
    days_in_coupon_period = (next_coupon_date - last_coupon_date).days

    if days_in_coupon_period == 0:
        return cleanPrice # Avoid division by zero

    accrued_interest = (100 * couponRate / 2) * (days_since_last_coupon / days_in_coupon_period)
    return cleanPrice + accrued_interest

# returns a list of tuples with [(coupon_date1, cf_amount1), ...] per $100 face value
def bondCashFlows(settlementDate, maturityDate, couponRate):

    bond_cash_flows = []
    coupon_dates = getCouponDates(settlementDate, maturityDate)

    for coupon_date in coupon_dates:
        bond_cash_flows.append((coupon_date, 100 * couponRate / 2))
    
    if not bond_cash_flows:
        # If no coupon dates, there is only a principal payment at maturity
        bond_cash_flows.append((maturityDate, 100))
        return bond_cash_flows

    # add back the principle to the last element of bond_cash_flows
    last_date, last_payment = bond_cash_flows[-1]
    bond_cash_flows[-1] = (last_date, last_payment + 100)

    return bond_cash_flows

#=======================MAIN PROGRAM=========================

# expects SettlementDate (string), Prices file path, CashFlows file path, and an output file path
if __name__ == "__main__":

    print("This is a module for dedicated portfolio construction.")
    if len(sys.argv) != 5:
        print("Usage: python dedicated_portfolio.py <settlement_date> <prices_csv> <cashflows.csv> <output_path>")
        print("Date format should be YYYY-MM-DD")
        sys.exit(1)
        
    settlement_date_str = str(sys.argv[1])
    prices_file_path = str(sys.argv[2])
    cashflows_file_path = str(sys.argv[3])
    output_file_path = str(sys.argv[4])

    try:
        settlement_date = pd.to_datetime(settlement_date_str).date()
        # Adjusted to handle the extra 'end of day' column
        prices_df = pd.read_csv(prices_file_path, header=None, names=['CUSIP', 'Security_Type', 'Rate', 'Maturity_Date', 'Call Date', 'Buy', 'Sell', 'End_of_Day'])
        prices_df['Maturity_Date'] = pd.to_datetime(prices_df['Maturity_Date']).dt.date
        cashflows_df = pd.read_csv(cashflows_file_path)
        cashflows_df.columns = ['Date', 'Liability']
        cashflows_df['Date'] = pd.to_datetime(cashflows_df['Date']).dt.date

    except FileNotFoundError as e:
        print(f"Error: Could not find an input file. Details: {e}")
        sys.exit(1)
    except ValueError:
        print(f"Error: Could not parse the date '{settlement_date_str}'. Please use the YYYY-MM-DD format.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        sys.exit(1)

    print(f"Settlement date: {settlement_date}")
    print(f"Using prices file: {prices_file_path}")
    print(f"Using cashflows file: {cashflows_file_path}")

    # =======================START OF INTERESTING CODE=========================

    # 1. --- Data Preparation and Filtering ---
    prices_df['Rate'] = pd.to_numeric(prices_df['Rate'], errors='coerce').fillna(0)
    prices_df['Buy'] = pd.to_numeric(prices_df['Buy'], errors='coerce').fillna(0)
    
    allowed_types = ['MARKET BASED BILL', 'MARKET BASED NOTE', 'MARKET BASED BOND']
    eligible_bonds = prices_df[prices_df['Security_Type'].isin(allowed_types)].copy()
    
    eligible_bonds = eligible_bonds[eligible_bonds['Maturity_Date'] > settlement_date]
    eligible_bonds = eligible_bonds[eligible_bonds['Buy'] > 0]
    eligible_bonds.reset_index(drop=True, inplace=True)
    
    cashflows_df = cashflows_df.sort_values(by='Date').reset_index(drop=True)

    # 2. --- Build Linear Programming Model ---
    
    # c: Cost vector (dirty price per $1 of face value for each bond)
    costs = []
    for _, bond in eligible_bonds.iterrows():
        is_bill = bond['Security_Type'] == 'MARKET BASED BILL'
        coupon_rate_decimal = bond['Rate'] / 100.0
        dp = dirtyPrice(settlement_date, bond['Maturity_Date'], coupon_rate_decimal, bond['Buy'], isbill=is_bill)
        costs.append(dp / 100.0)
    c = np.array(costs)

    # b_ub: Liabilities vector (negative of cumulative liabilities)
    b_ub = -cashflows_df['Liability'].cumsum().values

    # A_ub: Cash Flow matrix
    num_liabilities = len(cashflows_df)
    num_bonds = len(eligible_bonds)
    A = np.zeros((num_liabilities, num_bonds))
    liability_dates = cashflows_df['Date'].tolist()

    for i, (_, bond) in enumerate(eligible_bonds.iterrows()):
        cfs_per_dollar = []
        if bond['Security_Type'] == 'MARKET BASED BILL':
            cfs_per_dollar.append((bond['Maturity_Date'], 1.0))
        else:
            coupon_rate_decimal = bond['Rate'] / 100.0
            # Get cash flows per $100 and convert to per $1
            for cf_date, cf_amount in bondCashFlows(settlement_date, bond['Maturity_Date'], coupon_rate_decimal):
                cfs_per_dollar.append((cf_date, cf_amount / 100.0))
        
        cumulative_cf = 0
        cf_idx = 0
        for j, liability_date in enumerate(liability_dates):
            while cf_idx < len(cfs_per_dollar) and cfs_per_dollar[cf_idx][0] <= liability_date:
                cumulative_cf += cfs_per_dollar[cf_idx][1]
                cf_idx += 1
            A[j, i] = cumulative_cf
            
    # Constraint is Sum(CFs) >= Sum(Liabilities), which is equivalent to -Sum(CFs) <= -Sum(Liabilities)
    A_ub = -A

    # 3. --- Solve the Linear Program ---
    print("Solving optimization problem...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    
    # 4. --- Format the Output ---
    if result.success:
        print(f"Optimization successful. Minimum portfolio cost: ${result.fun:,.2f}")
        principals = result.x
        
        # Only include bonds where we purchase a meaningful amount (e.g., > $1)
        output_df = pd.DataFrame({
            'CUSIP': eligible_bonds['CUSIP'],
            'Principal': principals
        })
        output_df = output_df[output_df['Principal'] > 1.0]

    else:
        print("Error: Optimization failed. The problem may be infeasible.")
        print(f"Message: {result.message}")
        output_df = pd.DataFrame({'CUSIP': [], 'Principal': []})

    # =======================END OF INTERESTING CODE=========================

    # --- Create and save the output file ---
    try:
        output_df.to_csv(output_file_path, index=False)
        print(f"Output successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Error: Could not write to output file {output_file_path}. Details: {e}")
        sys.exit(1)