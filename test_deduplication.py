"""
Test script to verify deduplication functionality.
"""
import pandas as pd
import jellyfish
from src.models.data_models import DeduplicationColumnMap
from src.core.deduplication import build_duplicate_df

def main():
    print("Loading sample data...")
    df = pd.read_csv('sample_data.csv')
    print(f"Sample data loaded: {len(df)} rows, columns: {df.columns.tolist()}")
    
    # Create column mapping
    col_map = DeduplicationColumnMap(
        customer_name='Customer',
        address='Street', 
        city='City', 
        country='Country'
    )
    
    # Run deduplication
    print("Running deduplication...")
    results_df, stats = build_duplicate_df(
        df=df,
        col_map=col_map,
        use_prefix=True,
        use_metaphone=False,
        use_soundex=False,
        use_ngram=False,
        use_ai=False,
        name_threshold=70,
        overall_threshold=70
    )
    
    # Print results
    if not results_df.empty:
        total_dupes = results_df['DuplicateCount'].sum()
        print(f"Found {len(results_df)} master records with {total_dupes} total duplicates")
        
        # Print first few master records for verification
        print("\nSample master records:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            print(f"Master {i+1}: {row['MasterName']} - {row['DuplicateCount']} duplicates")
            # Print first 2 duplicates for this master
            for j, dup in enumerate(row['Duplicates'][:2]):
                print(f"  Duplicate {j+1}: {dup.get('Name')} - Score: {dup.get('Overall_score')}")
    else:
        print("No duplicates found")
    
    # Print block statistics
    print("\nBlocking statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 