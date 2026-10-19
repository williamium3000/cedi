"""
Scene Graph Data Structure

This module defines data structures for representing scene graphs,
including objects, relationships, and image information.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """Represents an object in the scene graph."""
    object_id: int
    x: int
    y: int
    w: int
    h: int
    names: List[str]
    synsets: List[str]
    attributes: List[str]
    
    def __post_init__(self):
        """Validate object data after initialization."""
        if self.w <= 0 or self.h <= 0:
            raise ValueError("Object width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("Object coordinates must be non-negative")
    def __repr__(self):
        return (f"Object(object_id={self.object_id}, x={self.x}, y={self.y}, w={self.w}, h={self.h}, "
                f"names={self.names}, synsets={self.synsets}, attributes={self.attributes})")


@dataclass
class Relationship:
    """Represents a relationship between two objects in the scene graph."""
    predicate: str
    synsets: str  # String representation of synset list
    subject: ObjectInfo
    object: ObjectInfo
    
    def get_synset_list(self) -> List[str]:
        """Parse the synsets string into a list."""
        import ast
        try:
            return ast.literal_eval(self.synsets)
        except (ValueError, SyntaxError):
            return []
    def __repr__(self):
        return (f"Relationship(predicate={self.predicate}, synsets={self.synsets}, subject={self.subject}, object={self.object})")


@dataclass
class SceneGraph:
    """Represents the scene graph structure with objects and relationships."""
    objects: Dict[str, ObjectInfo]  # Key is object identifier string
    relationships: List[Relationship]
    
    def get_object_by_id(self, object_id: int) -> Optional[ObjectInfo]:
        """Find an object by its numeric ID."""
        for obj in self.objects.values():
            if obj.object_id == object_id:
                return obj
        return None
    
    def get_objects_by_name(self, name: str) -> List[ObjectInfo]:
        """Find all objects with a specific name."""
        return [obj for obj in self.objects.values() if name in obj.names]
    
    def get_relationships_by_predicate(self, predicate: str) -> List[Relationship]:
        """Find all relationships with a specific predicate."""
        return [rel for rel in self.relationships if rel.predicate == predicate]

    def __repr__(self):
        ret = f"Objects:\n"
        for obj in self.objects.values():
            ret += f"{obj}\n"
        ret += "Relationships:\n"
        for rel in self.relationships:
            ret += f"{rel}\n"
        return ret

@dataclass
class ImageInfo:
    """Represents image metadata and associated scene graph."""
    image_id: int
    url: str = None  # Optional: can be None if image is a PIL object
    width: int = 0
    height: int = 0
    sg: SceneGraph = None
    
    def __post_init__(self):
        """Validate image data after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive")


@dataclass
class SceneGraphData:
    """Top-level container for scene graph data."""
    image_info: ImageInfo
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneGraphData':
        """Create SceneGraphData from a dictionary (e.g., loaded from JSON)."""
        img_info = data
        # Extract scene graph
        sg_data = img_info['sg']
        
        # Create objects
        objects = {}
        for obj_id, obj_data in sg_data['objects'].items():
            objects[obj_id] = ObjectInfo(
                object_id=obj_data['object_id'],
                x=obj_data['x'],
                y=obj_data['y'],
                w=obj_data['w'],
                h=obj_data['h'],
                names=obj_data['names'],
                synsets=obj_data['synsets'],
                attributes=obj_data['attributes']
            )
        
        # Create relationships
        relationships = []
        for rel_data in sg_data['relationships']:
            # Find subject and object
            subject_obj = None
            object_obj = None
            
            for obj in objects.values():
                if obj.object_id == rel_data['subject']['object_id']:
                    subject_obj = obj
                if obj.object_id == rel_data['object']['object_id']:
                    object_obj = obj
            
            if subject_obj and object_obj:
                relationships.append(Relationship(
                    predicate=rel_data['predicate'],
                    synsets=rel_data['synsets'],
                    subject=subject_obj,
                    object=object_obj
                ))
        
        # Create scene graph
        scene_graph = SceneGraph(objects=objects, relationships=relationships)
        
        # Create image info
        image_info = ImageInfo(
            image_id=img_info.get('image_id', 'unknown'),
            url=img_info.get('url', None),  # Optional for SVG with PIL images
            width=img_info.get('width', 640),
            height=img_info.get('height', 480),
            sg=scene_graph
        )
        
        return cls(image_info=image_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SceneGraphData to dictionary format."""
        objects_dict = {}
        for obj_id, obj in self.image_info.sg.objects.items():
            objects_dict[obj_id] = {
                'object_id': obj.object_id,
                'x': obj.x,
                'y': obj.y,
                'w': obj.w,
                'h': obj.h,
                'names': obj.names,
                'synsets': obj.synsets,
                'attributes': obj.attributes
            }
        
        relationships_list = []
        for rel in self.image_info.sg.relationships:
            relationships_list.append({
                'predicate': rel.predicate,
                'synsets': rel.synsets,
                'subject': {
                    'object_id': rel.subject.object_id,
                    'x': rel.subject.x,
                    'y': rel.subject.y,
                    'w': rel.subject.w,
                    'h': rel.subject.h,
                    'names': rel.subject.names,
                    'synsets': rel.subject.synsets,
                    'attributes': rel.subject.attributes
                },
                'object': {
                    'object_id': rel.object.object_id,
                    'x': rel.object.x,
                    'y': rel.object.y,
                    'w': rel.object.w,
                    'h': rel.object.h,
                    'names': rel.object.names,
                    'synsets': rel.object.synsets,
                    'attributes': rel.object.attributes
                }
            })
        
        return {
            'image_info': {
                'image_id': self.image_info.image_id,
                'url': self.image_info.url,
                'width': self.image_info.width,
                'height': self.image_info.height,
                'sg': {
                    'objects': objects_dict,
                    'relationships': relationships_list
                }
            }
        }


def load_scene_graph_data(json_file_path: str) -> List[SceneGraphData]:
    """Load scene graph data from a JSON file."""
    import json
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    return [SceneGraphData.from_dict(item) for item in data]


def save_scene_graph_data(scene_graphs: List[SceneGraphData], json_file_path: str) -> None:
    """Save scene graph data to a JSON file."""
    import json
    
    data = [sg.to_dict() for sg in scene_graphs]
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)


# Example usage and testing
if __name__ == "__main__":
    # Example of creating a scene graph programmatically
    obj1 = ObjectInfo(
        object_id=0,
        x=456, y=266, w=98, h=79,
        names=["office phone"],
        synsets=[],
        attributes=["multi-line phone"]
    )
    
    obj2 = ObjectInfo(
        object_id=1,
        x=83, y=202, w=17, h=24,
        names=["outlet"],
        synsets=["mercantile_establishment.n.01"],
        attributes=["electrical"]
    )
    
    # Create a relationship
    rel = Relationship(
        predicate="near",
        synsets="['near.r.01']",
        subject=obj1,
        object=obj2
    )
    
    # Create scene graph
    sg = SceneGraph(
        objects={"5091": obj1, "5092": obj2},
        relationships=[rel]
    )
    
    # Create image info
    img_info = ImageInfo(
        image_id=3,
        url="https://example.com/image.jpg",
        width=640,
        height=480,
        sg=sg
    )
    
    # Create top-level data structure
    scene_data = SceneGraphData(image_info=img_info)
    print(scene_data.image_info.sg)
    # print("Scene Graph Data Structure Created Successfully!")
    # print(f"Image ID: {scene_data.image_info.image_id}")
    # print(f"Number of objects: {len(scene_data.image_info.sg.objects)}")
    # print(f"Number of relationships: {len(scene_data.image_info.sg.relationships)}")
