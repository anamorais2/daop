def extended_gens(config, past_gen=False):
    config['epochs'] = config['extended_epochs']
    print(f"Extended gens mod: extended epochs to {config['epochs']}")

    if config['extended_pretrained_pretext_model'] is not None:
        config['pretrained_pretext_model'] = config['extended_pretrained_pretext_model']
        print(f"Extended gens mod: pretrained pretext model changed to extended version {config['pretrained_pretext_model']}")

    if not past_gen:
        config['start_population'] = [[individual[0], individual[1], None, None, None] for individual in config['best_individuals']]
        config['best_individuals'] = []
        print(f"Extended gens mod:  Resetting population to {config['start_population']}")