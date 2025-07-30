import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import networkx as nx
import numpy as np
from pathlib import Path
import random
import math

class RealDiagramRenderer:
    def __init__(self, output_dir="real_diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Color schemes for different domains
        self.color_schemes = {
            "ecommerce": {"table": "#e3f2fd", "border": "#1976d2", "text": "#0d47a1"},
            "hospital": {"table": "#f3e5f5", "border": "#7b1fa2", "text": "#4a148c"},
            "school": {"table": "#e8f5e8", "border": "#388e3c", "text": "#1b5e20"},
            "library": {"table": "#fff3e0", "border": "#f57c00", "text": "#e65100"},
            "crm": {"table": "#fce4ec", "border": "#c2185b", "text": "#880e4f"},
            "blog_platform": {"table": "#e0f2f1", "border": "#00695c", "text": "#004d40"},
            "default": {"table": "#f5f5f5", "border": "#424242", "text": "#212121"}
        }
    
    def calculate_optimal_layout(self, tables, relationships):
        """Calculate optimal table positions using network layout algorithms"""
        
        # Create a graph for layout calculation
        G = nx.Graph()
        
        # Add nodes (tables)
        for table in tables:
            G.add_node(table['name'])
        
        # Add edges (relationships)
        for rel in relationships:
            G.add_edge(rel['from_table'], rel['to_table'])
        
        # Use spring layout for better positioning
        if len(tables) <= 8:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        else:
            # For larger schemas, use hierarchical layout
            pos = nx.kamada_kawai_layout(G)
        
        # Convert to matplotlib coordinates
        positions = {}
        for table_name, (x, y) in pos.items():
            # Scale and translate to fit in figure
            positions[table_name] = (x * 4 + 5, y * 3 + 4)
        
        return positions
    
    def render_table(self, ax, table, position, color_scheme, max_width=2.5):
        """Render a single table with columns"""
        x, y = position
        table_name = table['name']
        columns = table['columns']
        
        # Calculate table dimensions
        row_height = 0.25
        header_height = 0.35
        table_height = header_height + len(columns) * row_height + 0.1
        table_width = min(max_width, max(2.0, len(table_name) * 0.12 + 0.5))
        
        # Draw table background
        table_rect = FancyBboxPatch(
            (x - table_width/2, y - table_height/2), 
            table_width, table_height,
            boxstyle="round,pad=0.02",
            facecolor=color_scheme["table"],
            edgecolor=color_scheme["border"],
            linewidth=1.5
        )
        ax.add_patch(table_rect)
        
        # Draw table header
        header_rect = FancyBboxPatch(
            (x - table_width/2 + 0.02, y + table_height/2 - header_height - 0.02), 
            table_width - 0.04, header_height,
            boxstyle="round,pad=0.01",
            facecolor=color_scheme["border"],
            edgecolor=color_scheme["border"],
            alpha=0.8
        )
        ax.add_patch(header_rect)
        
        # Table name
        ax.text(x, y + table_height/2 - header_height/2 - 0.02, 
               table_name.upper(), 
               ha='center', va='center', 
               fontsize=10, weight='bold', 
               color='white')
        
        # Draw separator line
        separator_y = y + table_height/2 - header_height - 0.02
        ax.plot([x - table_width/2 + 0.05, x + table_width/2 - 0.05], 
               [separator_y, separator_y], 
               color=color_scheme["border"], linewidth=1)
        
        # Draw columns
        for i, col in enumerate(columns):
            col_y = separator_y - (i + 1) * row_height + row_height/2
            
            # Column text
            col_text = col['name']
            if col.get('primary_key'):
                col_text = "🔑 " + col_text
            elif any(rel['from_table'] == table_name and rel['from_column'] == col['name'] 
                    for rel in []):  # Will be filled by caller
                col_text = "🔗 " + col_text
            
            # Data type
            col_type = col.get('type', '').upper()
            if len(col_type) > 15:
                col_type = col_type[:12] + "..."
            
            ax.text(x - table_width/2 + 0.1, col_y, 
                   col_text, 
                   ha='left', va='center', 
                   fontsize=8, 
                   color=color_scheme["text"],
                   weight='bold' if col.get('primary_key') else 'normal')
            
            ax.text(x + table_width/2 - 0.1, col_y, 
                   col_type, 
                   ha='right', va='center', 
                   fontsize=7, 
                   color=color_scheme["text"],
                   alpha=0.7,
                   style='italic')
        
        return {
            "bounds": (x - table_width/2, y - table_height/2, table_width, table_height),
            "center": (x, y)
        }
    
    def draw_relationship(self, ax, from_pos, to_pos, rel_type="many_to_one", color="#d32f2f"):
        """Draw relationship line between tables"""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Calculate connection points (edges of tables, not centers)
        # Simple approach: use centers for now, can be improved
        
        # Draw curved line
        if abs(from_x - to_x) > abs(from_y - to_y):
            # Horizontal curve
            mid_x = (from_x + to_x) / 2
            control1 = (from_x + (mid_x - from_x) * 0.3, from_y)
            control2 = (to_x - (to_x - mid_x) * 0.3, to_y)
        else:
            # Vertical curve
            mid_y = (from_y + to_y) / 2
            control1 = (from_x, from_y + (mid_y - from_y) * 0.3)
            control2 = (to_x, to_y - (to_y - mid_y) * 0.3)
        
        # Draw the relationship line
        ax.annotate('', xy=(to_x, to_y), xytext=(from_x, from_y),
                   arrowprops=dict(
                       arrowstyle='->' if rel_type == "many_to_one" else '-',
                       connectionstyle="arc3,rad=0.2",
                       color=color,
                       lw=1.5,
                       alpha=0.8
                   ))
    
    def render_er_diagram(self, schema_data, output_path, figsize=(14, 10)):
        """Render complete ER diagram from schema data"""
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        tables = schema_data['tables']
        relationships = schema_data.get('relationships', [])
        domain = schema_data.get('domain', 'default')
        
        # Get color scheme for domain
        color_scheme = self.color_schemes.get(domain, self.color_schemes['default'])
        
        # Calculate optimal layout
        positions = self.calculate_optimal_layout(tables, relationships)
        
        # Render tables
        table_info = {}
        for table in tables:
            table_name = table['name']
            if table_name in positions:
                pos = positions[table_name]
                table_bounds = self.render_table(ax, table, pos, color_scheme)
                table_info[table_name] = table_bounds
        
        # Draw relationships
        for rel in relationships:
            from_table = rel['from_table']
            to_table = rel['to_table']
            
            if from_table in table_info and to_table in table_info:
                from_pos = table_info[from_table]['center']
                to_pos = table_info[to_table]['center']
                self.draw_relationship(ax, from_pos, to_pos, rel.get('type', 'many_to_one'))
        
        # Add title and domain info
        title = f"Database Schema: {domain.replace('_', ' ').title()}"
        plt.suptitle(title, fontsize=16, weight='bold', y=0.95)
        
        # Add source info
        source = schema_data.get('source', 'unknown')
        ax.text(0.02, 0.02, f"Source: {source}", transform=ax.transAxes,
               fontsize=9, alpha=0.7, style='italic')
        
        # Add table count
        ax.text(0.98, 0.02, f"Tables: {len(tables)} | Relationships: {len(relationships)}", 
               transform=ax.transAxes, ha='right',
               fontsize=9, alpha=0.7, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def schema_to_sql(self, schema_data):
        """Convert schema to SQL DDL"""
        sql_statements = []
        
        # Create table statements
        for table in schema_data['tables']:
            columns = []
            primary_keys = []
            
            for col in table['columns']:
                col_def = f"{col['name']} {col['type']}"
                
                if col.get('primary_key'):
                    primary_keys.append(col['name'])
                
                if not col.get('nullable', True):
                    col_def += " NOT NULL"
                
                if col.get('unique'):
                    col_def += " UNIQUE"
                
                if col.get('default'):
                    default_val = col['default']
                    if isinstance(default_val, str) and not default_val.startswith("'"):
                        if default_val.upper() in ['CURRENT_TIMESTAMP', 'NOW()', 'TRUE', 'FALSE']:
                            col_def += f" DEFAULT {default_val}"
                        else:
                            col_def += f" DEFAULT '{default_val}'"
                    else:
                        col_def += f" DEFAULT {default_val}"
                
                columns.append(col_def)
            
            # Add primary key constraint if multiple columns
            if len(primary_keys) > 1:
                columns.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
            
            sql = f"CREATE TABLE {table['name']} (\n    " + ",\n    ".join(columns) + "\n);"
            sql_statements.append(sql)
        
        # Add foreign key constraints
        for rel in schema_data.get('relationships', []):
            if rel.get('type') in ['many_to_one', 'one_to_one']:
                fk_sql = (f"ALTER TABLE {rel['from_table']} "
                         f"ADD FOREIGN KEY ({rel['from_column']}) "
                         f"REFERENCES {rel['to_table']}({rel['to_column']});")
                sql_statements.append(fk_sql)
        
        return "\n\n".join(sql_statements)
    
    def process_all_schemas(self, schemas_file):
        """Process all schemas and create ER diagrams + SQL"""
        
        with open(schemas_file, 'r') as f:
            schemas = json.load(f)
        
        dataset = []
        
        print(f"Processing {len(schemas)} schemas...")
        
        for i, schema in enumerate(schemas):
            try:
                # Generate ER diagram
                image_path = self.output_dir / "images" / f"schema_{i:04d}.png"
                self.render_er_diagram(schema, image_path)
                
                # Generate SQL
                sql_output = self.schema_to_sql(schema)
                
                # Create training sample
                sample = {
                    "id": i,
                    "image": str(image_path),
                    "schema_json": schema,
                    "sql_output": sql_output,
                    "domain": schema.get('domain', 'unknown'),
                    "source": schema.get('source', 'unknown'),
                    "num_tables": len(schema['tables']),
                    "num_relationships": len(schema.get('relationships', []))
                }
                
                dataset.append(sample)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} schemas...")
            
            except Exception as e:
                print(f"Error processing schema {i}: {e}")
                continue
        
        # Save dataset
        dataset_file = self.output_dir / "training_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nDataset creation complete!")
        print(f"Generated {len(dataset)} training samples")
        print(f"Images saved to: {self.output_dir / 'images'}")
        print(f"Dataset saved to: {dataset_file}")
        
        # Print statistics
        sources = {}
        domains = {}
        for sample in dataset:
            source = sample['source']
            domain = sample['domain']
            sources[source] = sources.get(source, 0) + 1
            domains[domain] = domains.get(domain, 0) + 1
        
        print("\nDataset Statistics:")
        print("Sources:")
        for source, count in sources.items():
            print(f"  {source}: {count}")
        
        print("Domains:")
        for domain, count in domains.items():
            print(f"  {domain}: {count}")
        
        return dataset_file

# Integration script to combine everything
def create_real_schema_dataset():
    """Complete pipeline to create real schema dataset"""
    
    print("=== Creating Real Schema Dataset ===\n")
    
    # Step 1: Collect real schemas
    print("Step 1: Collecting real schemas...")
    from real_schema_collector import RealSchemaCollector
    
    collector = RealSchemaCollector()
    schema_file = collector.generate_all(
        num_programmatic=400,  # Generate 400 realistic programmatic schemas
        existing_db_connections=None,  # Add your DB connections here
        include_opensource=True
    )
    
    # Step 2: Generate ER diagrams and training data
    print("\nStep 2: Generating ER diagrams and training data...")
    renderer = RealDiagramRenderer()
    dataset_file = renderer.process_all_schemas(schema_file)
    
    print(f"\n✅ Complete! Training dataset ready at: {dataset_file}")
    return dataset_file

if __name__ == "__main__":
    dataset_file = create_real_schema_dataset()