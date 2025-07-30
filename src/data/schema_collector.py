import os
import json
import subprocess
import random
from typing import Dict, List, Any
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    Text, DateTime, Boolean, Float, ForeignKey, inspect
)
from sqlalchemy.schema import CreateTable
import requests
import re
from pathlib import Path

class RealSchemaCollector:
    def __init__(self, output_dir="real_schemas"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.schemas = []
    
    def method1_programmatic_generation(self, num_schemas=100):
        """Generate realistic schemas programmatically"""
        print("Generating programmatic schemas...")
        
        # Define realistic domain templates
        domain_templates = {
            "ecommerce": {
                "tables": ["users", "products", "orders", "order_items", "categories", "reviews"],
                "relationships": [
                    ("orders", "users", "user_id", "id"),
                    ("order_items", "orders", "order_id", "id"),
                    ("order_items", "products", "product_id", "id"),
                    ("products", "categories", "category_id", "id"),
                    ("reviews", "users", "user_id", "id"),
                    ("reviews", "products", "product_id", "id")
                ]
            },
            "hospital": {
                "tables": ["patients", "doctors", "appointments", "treatments", "departments", "medical_records"],
                "relationships": [
                    ("appointments", "patients", "patient_id", "id"),
                    ("appointments", "doctors", "doctor_id", "id"),
                    ("treatments", "appointments", "appointment_id", "id"),
                    ("doctors", "departments", "department_id", "id"),
                    ("medical_records", "patients", "patient_id", "id")
                ]
            },
            "school": {
                "tables": ["students", "teachers", "courses", "enrollments", "grades", "departments"],
                "relationships": [
                    ("enrollments", "students", "student_id", "id"),
                    ("enrollments", "courses", "course_id", "id"),
                    ("grades", "enrollments", "enrollment_id", "id"),
                    ("courses", "teachers", "teacher_id", "id"),
                    ("courses", "departments", "department_id", "id")
                ]
            },
            "library": {
                "tables": ["books", "authors", "members", "loans", "categories", "book_authors"],
                "relationships": [
                    ("loans", "books", "book_id", "id"),
                    ("loans", "members", "member_id", "id"),
                    ("book_authors", "books", "book_id", "id"),
                    ("book_authors", "authors", "author_id", "id"),
                    ("books", "categories", "category_id", "id")
                ]
            },
            "crm": {
                "tables": ["customers", "leads", "opportunities", "contacts", "companies", "activities"],
                "relationships": [
                    ("leads", "companies", "company_id", "id"),
                    ("opportunities", "customers", "customer_id", "id"),
                    ("contacts", "companies", "company_id", "id"),
                    ("activities", "customers", "customer_id", "id"),
                    ("activities", "contacts", "contact_id", "id")
                ]
            }
        }
        
        column_templates = {
            "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
            "created_at": {"type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
            "updated_at": {"type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
            "name": {"type": "VARCHAR(255)", "nullable": False},
            "email": {"type": "VARCHAR(255)", "unique": True},
            "phone": {"type": "VARCHAR(20)"},
            "address": {"type": "TEXT"},
            "description": {"type": "TEXT"},
            "price": {"type": "DECIMAL(10,2)"},
            "quantity": {"type": "INTEGER", "default": 0},
            "status": {"type": "VARCHAR(50)", "default": "'active'"},
            "is_active": {"type": "BOOLEAN", "default": "true"},
            "date_of_birth": {"type": "DATE"},
            "salary": {"type": "DECIMAL(12,2)"},
            "rating": {"type": "FLOAT"}
        }
        
        for i in range(num_schemas):
            domain = random.choice(list(domain_templates.keys()))
            template = domain_templates[domain]
            
            schema = {
                "domain": domain,
                "source": "programmatic",
                "tables": [],
                "relationships": []
            }
            
            # Generate tables with realistic columns
            for table_name in template["tables"]:
                table = {
                    "name": table_name,
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True}
                    ]
                }
                
                # Add domain-specific columns
                if table_name == "users" or table_name == "customers":
                    table["columns"].extend([
                        {"name": "email", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                        {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                        {"name": "phone", "type": "VARCHAR(20)"},
                        {"name": "created_at", "type": "TIMESTAMP"}
                    ])
                elif table_name == "products":
                    table["columns"].extend([
                        {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                        {"name": "description", "type": "TEXT"},
                        {"name": "price", "type": "DECIMAL(10,2)", "nullable": False},
                        {"name": "quantity", "type": "INTEGER", "default": 0}
                    ])
                elif table_name == "orders":
                    table["columns"].extend([
                        {"name": "order_date", "type": "TIMESTAMP", "nullable": False},
                        {"name": "total_amount", "type": "DECIMAL(10,2)"},
                        {"name": "status", "type": "VARCHAR(50)", "default": "'pending'"}
                    ])
                else:
                    # Add random columns for other tables
                    num_cols = random.randint(3, 7)
                    available_cols = list(column_templates.keys())
                    random.shuffle(available_cols)
                    
                    for col_name in available_cols[:num_cols]:
                        if col_name != "id":  # Already added
                            col_def = column_templates[col_name].copy()
                            table["columns"].append({"name": col_name, **col_def})
                
                schema["tables"].append(table)
            
            # Add relationships
            for from_table, to_table, from_col, to_col in template["relationships"]:
                # Add foreign key column to from_table
                from_table_obj = next((t for t in schema["tables"] if t["name"] == from_table), None)
                if from_table_obj and not any(c["name"] == from_col for c in from_table_obj["columns"]):
                    from_table_obj["columns"].append({
                        "name": from_col,
                        "type": "INTEGER",
                        "nullable": False
                    })
                
                schema["relationships"].append({
                    "from_table": from_table,
                    "to_table": to_table,
                    "from_column": from_col,
                    "to_column": to_col,
                    "type": "many_to_one"
                })
            
            self.schemas.append(schema)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1} programmatic schemas...")
    
    def method2_existing_databases(self, connection_strings: List[str]):
        """Extract schemas from existing databases using pg_dump and SQLAlchemy"""
        print("Extracting schemas from existing databases...")
        
        for i, conn_str in enumerate(connection_strings):
            try:
                # Method A: Using SQLAlchemy reflection
                engine = create_engine(conn_str)
                inspector = inspect(engine)
                
                schema = {
                    "domain": f"existing_db_{i}",
                    "source": "existing_database",
                    "connection": conn_str.split('@')[1] if '@' in conn_str else "unknown",
                    "tables": [],
                    "relationships": []
                }
                
                # Get all table names
                table_names = inspector.get_table_names()
                
                for table_name in table_names:
                    # Get columns
                    columns = inspector.get_columns(table_name)
                    table_columns = []
                    
                    for col in columns:
                        col_info = {
                            "name": col['name'],
                            "type": str(col['type']),
                            "nullable": col['nullable'],
                            "primary_key": col.get('primary_key', False)
                        }
                        if col.get('default'):
                            col_info['default'] = str(col['default'])
                        table_columns.append(col_info)
                    
                    schema["tables"].append({
                        "name": table_name,
                        "columns": table_columns
                    })
                
                # Get foreign keys
                for table_name in table_names:
                    fks = inspector.get_foreign_keys(table_name)
                    for fk in fks:
                        schema["relationships"].append({
                            "from_table": table_name,
                            "to_table": fk['referred_table'],
                            "from_column": fk['constrained_columns'][0],
                            "to_column": fk['referred_columns'][0],
                            "type": "many_to_one"
                        })
                
                self.schemas.append(schema)
                print(f"Extracted schema from database {i + 1}")
                
            except Exception as e:
                print(f"Error extracting from database {i + 1}: {e}")
                continue
    
    def method3_opensource_apps(self):
        """Extract schemas from open-source applications"""
        print("Extracting schemas from open-source apps...")
        
        # GitHub repositories with SQL schema files
        repos_with_schemas = [
            {
                "name": "odoo",
                "files": [
                    "https://raw.githubusercontent.com/odoo/odoo/17.0/addons/base/data/res_partner_demo.xml",
                    "https://raw.githubusercontent.com/odoo/odoo/17.0/odoo/addons/base/models/res_partner.py"
                ]
            },
            {
                "name": "supabase",
                "files": [
                    "https://raw.githubusercontent.com/supabase/supabase/master/examples/nextjs-todo-list/supabase/migrations/20220127152649_init.sql"
                ]
            },
            {
                "name": "gitlab",
                "files": [
                    "https://raw.githubusercontent.com/gitlabhq/gitlabhq/master/db/structure.sql"
                ]
            }
        ]
        
        # Common SQL schema patterns from various open source projects
        common_schemas = {
            "blog_platform": {
                "domain": "blog_platform",
                "source": "opensource_app",
                "tables": [
                    {
                        "name": "users",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "username", "type": "VARCHAR(50)", "unique": True, "nullable": False},
                            {"name": "email", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                            {"name": "password_hash", "type": "VARCHAR(255)", "nullable": False},
                            {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
                            {"name": "updated_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "posts",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "title", "type": "VARCHAR(255)", "nullable": False},
                            {"name": "slug", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                            {"name": "content", "type": "TEXT", "nullable": False},
                            {"name": "author_id", "type": "INTEGER", "nullable": False},
                            {"name": "published", "type": "BOOLEAN", "default": False},
                            {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
                            {"name": "updated_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "comments",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "post_id", "type": "INTEGER", "nullable": False},
                            {"name": "author_id", "type": "INTEGER", "nullable": False},
                            {"name": "content", "type": "TEXT", "nullable": False},
                            {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "tags",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)", "unique": True, "nullable": False},
                            {"name": "slug", "type": "VARCHAR(100)", "unique": True, "nullable": False}
                        ]
                    },
                    {
                        "name": "post_tags",
                        "columns": [
                            {"name": "post_id", "type": "INTEGER", "nullable": False},
                            {"name": "tag_id", "type": "INTEGER", "nullable": False}
                        ]
                    }
                ],
                "relationships": [
                    {"from_table": "posts", "to_table": "users", "from_column": "author_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "comments", "to_table": "posts", "from_column": "post_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "comments", "to_table": "users", "from_column": "author_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "post_tags", "to_table": "posts", "from_column": "post_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "post_tags", "to_table": "tags", "from_column": "tag_id", "to_column": "id", "type": "many_to_one"}
                ]
            },
            "ecommerce_platform": {
                "domain": "ecommerce_platform", 
                "source": "opensource_app",
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "first_name", "type": "VARCHAR(100)", "nullable": False},
                            {"name": "last_name", "type": "VARCHAR(100)", "nullable": False},
                            {"name": "email", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                            {"name": "phone", "type": "VARCHAR(20)"},
                            {"name": "date_of_birth", "type": "DATE"},
                            {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "addresses",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "customer_id", "type": "INTEGER", "nullable": False},
                            {"name": "type", "type": "VARCHAR(20)", "nullable": False},
                            {"name": "street_address", "type": "VARCHAR(255)", "nullable": False},
                            {"name": "city", "type": "VARCHAR(100)", "nullable": False},
                            {"name": "state", "type": "VARCHAR(100)"},
                            {"name": "postal_code", "type": "VARCHAR(20)", "nullable": False},
                            {"name": "country", "type": "VARCHAR(100)", "nullable": False}
                        ]
                    },
                    {
                        "name": "categories", 
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                            {"name": "slug", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                            {"name": "description", "type": "TEXT"},
                            {"name": "parent_id", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "products",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                            {"name": "slug", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                            {"name": "description", "type": "TEXT"},
                            {"name": "price", "type": "DECIMAL(10,2)", "nullable": False},
                            {"name": "cost", "type": "DECIMAL(10,2)"},
                            {"name": "sku", "type": "VARCHAR(100)", "unique": True},
                            {"name": "stock_quantity", "type": "INTEGER", "default": 0},
                            {"name": "category_id", "type": "INTEGER", "nullable": False},
                            {"name": "is_active", "type": "BOOLEAN", "default": True},
                            {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "orders",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "customer_id", "type": "INTEGER", "nullable": False},
                            {"name": "order_number", "type": "VARCHAR(50)", "unique": True, "nullable": False},
                            {"name": "status", "type": "VARCHAR(50)", "default": "'pending'"},
                            {"name": "subtotal", "type": "DECIMAL(10,2)", "nullable": False},
                            {"name": "tax_amount", "type": "DECIMAL(10,2)", "default": 0},
                            {"name": "shipping_amount", "type": "DECIMAL(10,2)", "default": 0},
                            {"name": "total_amount", "type": "DECIMAL(10,2)", "nullable": False},
                            {"name": "order_date", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
                            {"name": "shipping_address_id", "type": "INTEGER"},
                            {"name": "billing_address_id", "type": "INTEGER"}
                        ]
                    },
                    {
                        "name": "order_items",
                        "columns": [
                            {"name": "id", "type": "INTEGER", "primary_key": True},
                            {"name": "order_id", "type": "INTEGER", "nullable": False},
                            {"name": "product_id", "type": "INTEGER", "nullable": False},
                            {"name": "quantity", "type": "INTEGER", "nullable": False},
                            {"name": "unit_price", "type": "DECIMAL(10,2)", "nullable": False},
                            {"name": "total_price", "type": "DECIMAL(10,2)", "nullable": False}
                        ]
                    }
                ],
                "relationships": [
                    {"from_table": "addresses", "to_table": "customers", "from_column": "customer_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "categories", "to_table": "categories", "from_column": "parent_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "products", "to_table": "categories", "from_column": "category_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "orders", "to_table": "customers", "from_column": "customer_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "orders", "to_table": "addresses", "from_column": "shipping_address_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "orders", "to_table": "addresses", "from_column": "billing_address_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "order_items", "to_table": "orders", "from_column": "order_id", "to_column": "id", "type": "many_to_one"},
                    {"from_table": "order_items", "to_table": "products", "from_column": "product_id", "to_column": "id", "type": "many_to_one"}
                ]
            }
        }
        
        # Add these realistic schemas
        for schema_name, schema_data in common_schemas.items():
            self.schemas.append(schema_data)
            print(f"Added {schema_name} schema")
    
    def save_schemas(self, filename="real_schemas.json"):
        """Save all collected schemas to JSON file"""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(self.schemas, f, indent=2)
        
        print(f"Saved {len(self.schemas)} schemas to {output_file}")
        return output_file
    
    def generate_all(self, 
                     num_programmatic=200,
                     existing_db_connections=None,
                     include_opensource=True):
        """Generate schemas using all three methods"""
        
        print("Starting real schema collection...")
        
        # Method 1: Programmatic generation
        self.method1_programmatic_generation(num_programmatic)
        
        # Method 2: Existing databases (if provided)
        if existing_db_connections:
            self.method2_existing_databases(existing_db_connections)
        
        # Method 3: Open source apps
        if include_opensource:
            self.method3_opensource_apps()
        
        # Save all schemas
        schema_file = self.save_schemas()
        
        print(f"\nCollection complete!")
        print(f"Total schemas: {len(self.schemas)}")
        print(f"Sources breakdown:")
        sources = {}
        for schema in self.schemas:
            source = schema['source']
            sources[source] = sources.get(source, 0) + 1
        for source, count in sources.items():
            print(f"  {source}: {count}")
        
        return schema_file

# Usage example
if __name__ == "__main__":
    collector = RealSchemaCollector()
    
    # Example database connections (replace with real ones if available)
    # existing_connections = [
    #     "postgresql://user:password@localhost:5432/mydb",
    #     "mysql://user:password@localhost:3306/mydb"
    # ]
    
    schema_file = collector.generate_all(
        num_programmatic=300,
        existing_db_connections=None,  # Add your DB connections here
        include_opensource=True
    )
    
    print(f"Real schemas saved to: {schema_file}")