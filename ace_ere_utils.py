class Token(object):
    def __init__(self, id_, text, start, end):
        self.id_ = id_
        self.text = text
        self.start = start
        self.end = end

    ## Setters
    def set_id_(self, id_):
        self.id_ = id_

    def set_text(self, text):
        self.text = text

    def set_start(self, start):
        self.start = start

    def set_end(self, end):
        self.end = end

    ## Getters
    def get_id_(self):
        return self.id_

    def get_text(self):
        return self.text

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def to_string(self):
        return "Token: {id_ = " + self.id_ + ", text = " + self.text + ", start = " + self.start + ", end = " \
               + self.end  +  "}"


class Sentence(object):
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end

    def to_string(self):
        return "Sentence: {text = " + self.text + ", start = " + self.start + ", end = " + self.end + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class ACEDocument(object):
    def __init__(self, id_, source, datetime, text):
        self.id_ = id_
        self.source = source
        self.datetime = datetime
        self.text = text

    def to_string(self):
        text_str = "["
        for t in self.text:
            text_str += t + ","
        text_str += "]"
        return "Document: {id_ = " + self.id_ + ", source = " + self.source + ", datetime = " + self.datetime  + ", text = " + text_str + "}"


class Entity(object):
    def __init__(self, id_, text, entity_type, phrase_type, start, end):
        self.id_ = id_
        self.text = text
        self.entity_type = entity_type
        self.phrase_type = phrase_type
        self.start = start
        self.end = end

    def to_string(self):
        return "Entity: {id_ = " + self.id_ + ", text = " + self.text + ", entity_type = " + self.entity_type + ", phrase_type=" + self.phrase_type + ", start =" + str(self.start) + ", end =" + str(self.end) + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class EREEntity(object):
    def __init__(self, ent_id, ent_type, ent_specif, ent_mention_id, noun_type, source, entity_start, entity_end, mention_text):
        self.ent_id = ent_id
        self.ent_type = ent_type
        self.ent_specif = ent_specif
        self.ent_mention_id = ent_mention_id
        self.noun_type = noun_type
        self.source = source
        self.entity_start = entity_start
        self.entity_end = entity_end
        self.mention_text = mention_text

    ### Setters
    def set_ent_id(self, ent_id):
        self.ent_id = ent_id

    def set_ent_type(self, ent_type):
        self.ent_type = ent_type

    def set_ent_specif(self, ent_specif):
        self.ent_specif = ent_specif

    def set_ent_mention_id(self, ent_mention_id):
        self.ent_mention_id = ent_mention_id

    def set_noun_type(self, noun_type):
        self.noun_type = noun_type

    def set_source(self, source):
        self.source = source

    def set_entity_start(self, entity_start):
        self.entity_start = entity_start

    def set_entity_end(self, entity_end):
        self.entity_end = entity_end

    def set_mention_text(self, mention_text):
        self.mention_text = mention_text

    ### Getters
    def get_ent_id(self):
        return self.ent_id

    def get_ent_type(self):
        return self.ent_type

    def get_ent_specif(self):
        return self.ent_specif

    def get_ent_mention_id(self):
        return self.ent_mention_id

    def get_noun_type(self):
        return self.noun_type

    def get_source(self):
        return self.source

    def get_entity_start(self):
        return self.entity_start

    def get_entity_end(self):
        return self.entity_end

    def get_mention_text(self):
        return self.mention_text


class Filler(object):
    def __init__(self, fill_id, fill_src, fill_st, fill_ed, fill_type):
        self.fill_id = fill_id
        self.fill_src = fill_src
        self.fill_st = fill_st
        self.fill_ed = fill_ed
        self.fill_type = fill_type

    ### Setters
    def set_fill_id(self, fill_id):
        self.fill_id = fill_id

    def set_fill_src(self, fill_src):
        self.fill_src = fill_src

    def set_fill_st(self, fill_start):
        self.fill_st = fill_start

    def set_fill_end(self, fill_end):
        self.fill_ed = fill_end

    def set_fill_type(self, fill_type):
        self.fill_type = fill_type

    ### Getters
    def get_fill_id(self):
        return self.fill_id

    def get_fill_src(self):
        return self.fill_src

    def get_fill_st(self):
        return self.fill_st

    def get_fill_ed(self):
        return self.fill_ed

    def get_fill_type(self):
        return self.fill_type


class Trigger(object):
    def __init__(self, start, text, end, id_, event_type):
        self.start = start
        self.text = text
        self.end = end
        self.id_ = id_
        self.event_type = event_type

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Argument(object):
    def __init__(self, id_, text, role, start, end, entity_type):
        self.id_ = id_
        self.text = text
        self.role = role
        self.start = start
        self.end = end
        self.entity_type = entity_type

    def to_string(self):
        return "Argument: {id_ = " + self.id_ + ", text = " + self.text + ", role = " + self.role + ", start =" + str(self.start) + ", end =" + str(self.end) + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Event(object):
    def __init__(self, event_id, mention_id, type_, subtype, modality, polarity, genericity, tense, extent, extent_start, extent_end, scope, scope_start, scope_end, trig_text, trig_start, trig_end, arguments, entities):
        self.event_id = event_id
        self.mention_id = mention_id
        self.type_ = type_
        self.subtype = subtype
        self.modality = modality
        self.polarity = polarity
        self.genericity = genericity
        self.tense = tense
        self.extent = extent
        self.extent_start = extent_start
        self.extent_end = extent_end
        self.scope = scope
        self.scope_start = scope_start
        self.scope_end = scope_end
        self.trig_text = trig_text
        self.trig_start = trig_start
        self.trig_end = trig_end
        self.arguments = arguments
        self.entities = entities

    def to_string(self):
        return "Event: { event_id = " + self.event_id + "mention_id = " + self.mention_id + ", type = " + self.type_ + ", subtype = " +self.subtype + ", modality = " \
               + self.modality + ", polarity = " + self.polarity + ", genericity= " + self.genericity + ", tense = " + \
               self.tense + ", extent = " + self.extent + ", scope = " + self.scope  + ", trigger = " + self.trig_text

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)
