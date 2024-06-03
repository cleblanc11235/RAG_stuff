import yaml
from collections import OrderedDict

def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

#TODO - Can generalize using **kwargs
def YAMLize(model, date, expansion, prompts, prompts_outputs, supports, args):
    purdy_date = f"{date[:2]}-{date[2:4]}-{date[4:]}"
    test_data = OrderedDict([
        ('model', model),
        ('date', purdy_date),
        ('Query Expansion', expansion),
        ('prompts', prompts),
        ('arguments', {
            'doc_path': args.doc_path,
            'client_path': args.client_path,
            'collection_name': args.collection_name,
            'k': args.k,
            'force_reload': args.REFRESH_DOC_EMBEDDING,
            'temperature': args.temperature,
            'threshold': args.threshold,
            'no_expand': args.no_expand,
            'pool_responses': args.pool_responses,
            'n_multishot': args.n_multishot,
            'viewport_width': args.viewport_width,
            'SHOW_OUTPUTS': args.SHOW_OUTPUTS
        })
    ])

    # Process 'prompts_outputs' and set up 'answers'
    answers = OrderedDict()
    for idx, outputs in enumerate(prompts_outputs, start=1):
        prompt_id = f'q{idx}'
        answer_supports = supports[idx-1] if idx-1 < len(supports) else []
        answer_details = OrderedDict()
        for answer_idx, answer in enumerate(outputs):
            answer_details[f'answer{answer_idx + 1}'] = {
                'response': answer,
                'support': answer_supports[answer_idx] if answer_idx < len(answer_supports) else []
            }
        answers[prompt_id] = answer_details
    
    test_data['answers'] = answers
    return test_data
