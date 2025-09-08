def process_all_vision_info_for_nuextract(messages, examples=None):
    """
    Process vision information from both messages and in-context examples, supporting batch processing.
    
    Args:
        messages: List of message dictionaries (single input) OR list of message lists (batch input)
        examples: Optional list of example dictionaries (single input) OR list of example lists (batch)
    
    Returns:
        A flat list of all images in the correct order:
        - For single input: example images followed by message images
        - For batch input: interleaved as (item1 examples, item1 input, item2 examples, item2 input, etc.)
        - Returns None if no images were found
    """
    from qwen_vl_utils import process_vision_info, fetch_image
    
    # Helper function to extract images from examples
    def extract_example_images(example_item):
        if not example_item:
            return []
            
        # Handle both list of examples and single example
        examples_to_process = example_item if isinstance(example_item, list) else [example_item]
        images = []
        
        for example in examples_to_process:
            if isinstance(example.get('input'), dict) and example['input'].get('type') == 'image':
                images.append(fetch_image(example['input']))
                
        return images
    
    # Normalize inputs to always be batched format
    is_batch = messages and isinstance(messages[0], list)
    messages_batch = messages if is_batch else [messages]
    is_batch_examples = examples and isinstance(examples, list) and (isinstance(examples[0], list) or examples[0] is None)
    examples_batch = examples if is_batch_examples else ([examples] if examples is not None else None)
    
    # Ensure examples batch matches messages batch if provided
    if examples and len(examples_batch) != len(messages_batch):
        if not is_batch and len(examples_batch) == 1:
            # Single example set for a single input is fine
            pass
        else:
            raise ValueError("Examples batch length must match messages batch length")
    
    # Process all inputs, maintaining correct order
    all_images = []
    for i, message_group in enumerate(messages_batch):
        # Get example images for this input
        if examples and i < len(examples_batch):
            input_example_images = extract_example_images(examples_batch[i])
            all_images.extend(input_example_images)
        
        # Get message images for this input
        input_message_images = process_vision_info(message_group)[0] or []
        all_images.extend(input_message_images)
    
    return all_images if all_images else None
