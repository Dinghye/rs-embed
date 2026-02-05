# **🔌 Extending the Toolkit**

1. Create a new embedder:
```
@register("my_model")
class MyEmbedder(EmbedderBase):
    def get_embedding(...):
        ...
```    
2. Return a unified Embedding(data, meta)
3. Automatically available via get_embedding(...)

> You only need to implement:
> **ROI → embedding**
> Everything else (specs, metadata, output formatting) is handled by the framework.