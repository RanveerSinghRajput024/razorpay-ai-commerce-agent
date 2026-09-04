import pandas as pd


def load_transactions(file_path):
    """Load the transaction dataset."""
    return pd.read_excel(file_path)


def clean_transactions(df):
    """Clean transaction data for analysis."""

    df = df.copy()

    # Remove cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove invalid prices
    df = df[df["UnitPrice"] > 0]

    # Remove invalid quantities
    df = df[df["Quantity"] > 0]

    # Calculate revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)


def load_product_relationships(file_path):
    """Load the final product relationship data."""
    return pd.read_csv(file_path)