#!/usr/bin/env python3
"""
Script to generate Mermaid diagrams from schema JSON files
"""

import json
import os

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
    
    print(f"Generated Mermaid diagram: {output_file}")

def main():
    # Read the schemas file
    schemas_file = "data/real_schemas/real_schemas.json"
    
    with open(schemas_file, 'r', encoding='utf-8') as f:
        schemas = json.load(f)
    
    # Generate for the first schema (schema_0000)
    first_schema = schemas[0]
    output_file = "schema_0000.mmd"
    
    generate_mermaid_erd(first_schema, output_file)
    
    print(f"Domain: {first_schema['domain']}")
    print(f"Number of tables: {len(first_schema['tables'])}")
    print(f"Number of relationships: {len(first_schema.get('relationships', []))}")

if __name__ == "__main__":
    main()
