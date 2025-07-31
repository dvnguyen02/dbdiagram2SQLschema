#!/usr/bin/env python3
"""
Script to batch generate Mermaid diagrams from all schemas in the JSON file
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def generate_mermaid_erd(schema_data, output_file):
    """Generate Mermaid ERD diagram from schema data"""
    
    mermaid_content = ["erDiagram"]
    
    # Generate table definitions
    for table in schema_data["tables"]:
        table_name = table["name"]
        mermaid_content.append(f"    {table_name} {{")
        
        for column in table["columns"]:
            col_name = column["name"]
            col_type = column["type"]
            
            # Clean up column type to avoid parsing issues
            # Remove parentheses and special characters that might cause issues
            clean_type = col_type.replace("(", "_").replace(")", "").replace(",", "_")
            
            # Add constraints/attributes
            attributes = []
            if column.get("primary_key"):
                attributes.append("PK")
            if column.get("unique"):
                attributes.append("UK")
            if column.get("nullable") == False:
                attributes.append("NOT NULL")
            
            if attributes:
                attr_str = f" \"{', '.join(attributes)}\""
            else:
                attr_str = ""
                
            mermaid_content.append(f"        {clean_type} {col_name}{attr_str}")
        
        mermaid_content.append("    }")
        mermaid_content.append("")
    
    # Generate relationships
    if "relationships" in schema_data:
        for rel in schema_data["relationships"]:
            from_table = rel["from_table"]
            to_table = rel["to_table"]
            rel_type = rel["type"]
            
            # Convert relationship type to Mermaid notation
            if rel_type == "many_to_one":
                relation_symbol = "}o--||"
            elif rel_type == "one_to_many":
                relation_symbol = "||--o{"
            elif rel_type == "one_to_one":
                relation_symbol = "||--||"
            elif rel_type == "many_to_many":
                relation_symbol = "}o--o{"
            else:
                relation_symbol = "--"
            
            mermaid_content.append(f"    {from_table} {relation_symbol} {to_table} : \"{rel['from_column']} -> {rel['to_column']}\"")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mermaid_content))
    
    return True

def convert_to_png(mmd_file, png_file):
    """Convert mermaid file to PNG using mmdc"""
    # Try different mmdc command locations
    mmdc_commands = [
        'mmdc',  # If in PATH
        'C:\\Users\\ASUS\\AppData\\Roaming\\npm\\mmdc.cmd',  # Windows npm global
        '/usr/local/bin/mmdc',  # Common macOS/Linux location
    ]
    
    for mmdc_cmd in mmdc_commands:
        try:
            result = subprocess.run([mmdc_cmd, '-i', str(mmd_file), '-o', str(png_file)], 
                                  capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print(f"Error: mmdc command not found in any of the expected locations")
    print("Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
    return False

def main():
    # Read the schemas file
    schemas_file = "data/real_schemas/real_schemas.json"
    
    if not os.path.exists(schemas_file):
        print(f"Error: {schemas_file} not found")
        return
    
    with open(schemas_file, 'r', encoding='utf-8') as f:
        schemas = json.load(f)
    
    print(f"Found {len(schemas)} schemas to process")
    
    # Create output directories for mermaid files and images
    mmd_dir = Path("generated_diagrams/mermaid")
    png_dir = Path("generated_diagrams/images")
    mmd_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    
    # Process command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # Process all schemas
            start_idx = 0
            end_idx = len(schemas)
        else:
            try:
                # Process specific schema by index
                schema_idx = int(sys.argv[1])
                if schema_idx < 0 or schema_idx >= len(schemas):
                    print(f"Error: Schema index {schema_idx} out of range (0-{len(schemas)-1})")
                    return
                start_idx = schema_idx
                end_idx = schema_idx + 1
            except ValueError:
                print("Usage: python generate_all_mermaid.py [schema_index|all]")
                print("Example: python generate_all_mermaid.py 0  # Generate only schema_0000")
                print("Example: python generate_all_mermaid.py all  # Generate all schemas")
                return
    else:
        # Default: process first 5 schemas
        start_idx = 0
        end_idx = min(5, len(schemas))
        print(f"Processing first {end_idx} schemas (use 'all' argument to process all)")
    
    success_count = 0
    failed_count = 0
    
    for i in range(start_idx, end_idx):
        schema = schemas[i]
        schema_id = f"schema_{i:04d}"
        
        # Skip schemas without table data
        if "tables" not in schema or not schema["tables"]:
            print(f"Skipping {schema_id}: No table data")
            continue
        
        mmd_file = mmd_dir / f"{schema_id}.mmd"
        png_file = png_dir / f"{schema_id}.png"
        
        print(f"Processing {schema_id}...")
        
        # Generate mermaid file
        try:
            generate_mermaid_erd(schema, mmd_file)
            print(f"  ✓ Generated {mmd_file}")
        except Exception as e:
            print(f"  ✗ Failed to generate mermaid file: {e}")
            failed_count += 1
            continue
        
        # Convert to PNG
        if convert_to_png(mmd_file, png_file):
            print(f"  ✓ Generated {png_file}")
            success_count += 1
        else:
            print(f"  ✗ Failed to convert to PNG")
            failed_count += 1
        
        # Print schema info
        domain = schema.get('domain', 'unknown')
        num_tables = len(schema.get('tables', []))
        num_relationships = len(schema.get('relationships', []))
        print(f"    Domain: {domain}, Tables: {num_tables}, Relationships: {num_relationships}")
    
    print(f"\nCompleted: {success_count} successful, {failed_count} failed")

if __name__ == "__main__":
    main()
